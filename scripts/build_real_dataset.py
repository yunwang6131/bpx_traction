#!/usr/bin/env python3
"""把真机 v6 录制打成微调用的数据集。

    python scripts/build_real_dataset.py --rec logs/real_v6/rec_20260418_191052 \
                                         --out logs/traction/real_finetune

真机上没有力传感器，但这些录制里有**两个免费标签**：

* **外力 = 0** —— 采集全程没人碰机器人，这条是确定的，对**每一帧**都成立。
* **机身速度 ≈ 指令低通后的值** —— 弱标签。策略在仿真里的稳态跟踪误差是
  `metrics.POLICY_TRACKING_ERROR_MPS`（0.053 m/s），但指令刚跳变时实际速度还在爬，
  这时 raw cmd 和真速度差很远。所以先把指令过一个一阶低通再当标签。

  时间常数是实测的：真机上 cmd 和估计器速度头的互相关在**滞后 23 帧 = 460 ms** 处
  最高（0.874）；低通 τ 从 0 扫到 600 ms，相关从 0.587 升到 0.837，400 ms 之后
  基本饱和，所以取 ``--tau-ms 400``。相关是尺度无关的，速度头本身的幅值偏差
  （真机低估 61%）不影响这个估计 —— 它量的是时序对不对得上。

  指令稳住一段时间的帧标签更可信，给满权重；瞬态帧权重打折，由 ``--steady-s`` 和
  ``--transient-weight`` 控制。

为什么要第二个标签：真机误报力的慢漂移和"估计速度 − 指令速度"的相关系数是 +0.759，
是所有候选里最强的 —— 力头在吃速度头的残差。只压力是打症状，把速度头掰正才是对因。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from bpx_traction import spec  # noqa: E402
from real_deadzone import build_frames  # noqa: E402


def lowpass(cmd: np.ndarray, tau_ms: float) -> np.ndarray:
    """一阶低通，近似"实际速度怎么跟上指令"。零阶保持的精确离散。"""
    if tau_ms <= 0:
        return cmd.copy()
    alpha = float(np.exp(-spec.SAMPLE_PERIOD_S * 1000.0 / tau_ms))
    out = np.empty_like(cmd)
    out[0] = cmd[0] * (1.0 - alpha)      # 从静止起步，不要让第一帧直接等于指令
    for i in range(1, len(cmd)):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * cmd[i]
    return out


def steady_mask(cmd: np.ndarray, steady_s: float, tol: float = 0.02) -> np.ndarray:
    """指令连续 `steady_s` 秒没变过的帧 —— 只有这些帧上 cmd ≈ 实际速度。"""
    n = int(round(steady_s / spec.SAMPLE_PERIOD_S))
    if n <= 1:
        return np.ones(len(cmd), dtype=bool)
    changed = np.r_[True, (np.abs(np.diff(cmd, axis=0)) > tol).any(axis=1)]
    # 距离上一次变化有多少帧
    since = np.zeros(len(cmd), dtype=np.int32)
    k = 0
    for i, c in enumerate(changed):
        k = 0 if c else k + 1
        since[i] = k
    return since >= n


def coverage_report(hist, cmd, src, names) -> None:
    """数据覆盖审计：微调能不能泛化，取决于这几个轴上采够没有。

    只报数不判定 —— 多少算够取决于目标。但**缺一整个轴**（比如全是平地直行、
    没有转向）一眼就能看出来，而那正是现在过拟合的原因：一次录制、一种地面。

    俯仰这一栏值得单独盯：真机误报和 `|重力x|` 的相关是 +0.615，按姿态分层能看出
    误报是不是集中在姿态变化大的时候。
    """
    last = hist[:, -1]
    g = last[:, slice(*spec.term_slice("projected_gravity_b"))]
    # 重力在机身系的 −z 方向，倾角 = 它和 −ẑ 的夹角
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2] / np.linalg.norm(g, axis=1), -1.0, 1.0)))
    vx, vy, wz = (np.abs(cmd[:, k]) for k in range(3))
    sec = lambda m: m.sum() / spec.POLICY_HZ

    print("\n[覆盖审计]")
    print(f"  {'录制':<34}{'秒':>7}{'走路':>8}{'转向':>8}{'侧移':>8}")
    for k, name in enumerate(names):
        m = src == k
        print(f"  {Path(name).name:<34}{sec(m):>7.0f}{sec(m & (vx > 0.3)):>8.0f}"
              f"{sec(m & (wz > 0.3)):>8.0f}{sec(m & (vy > 0.1)):>8.0f}")
    print(f"  {'合计':<34}{sec(np.ones(len(src), bool)):>7.0f}{sec(vx > 0.3):>8.0f}"
          f"{sec(wz > 0.3):>8.0f}{sec(vy > 0.1):>8.0f}")

    print(f"\n  前进速度分档(s)   " + "  ".join(
        f"{lo}–{hi}:{sec((vx >= lo) & (vx < hi)):.0f}"
        for lo, hi in ((0.05, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 99))))
    print(f"  机身倾角分档(s)   " + "  ".join(
        f"{lo}–{hi}°:{sec((tilt >= lo) & (tilt < hi)):.0f}"
        for lo, hi in ((0, 5), (5, 10), (10, 20), (20, 180))))
    print(f"  倾角 p50 {np.percentile(tilt, 50):.1f}°  p95 {np.percentile(tilt, 95):.1f}°  "
          f"max {tilt.max():.1f}°")

    gaps = []
    if len(names) < 3:
        gaps.append(f"只有 {len(names)} 段独立录制 —— 样本外指标会被单次 session 的特性带偏")
    if sec(wz > 0.3) < 30:
        gaps.append("转向不足 30 s")
    if sec(vy > 0.1) < 20:
        gaps.append("侧移不足 20 s")
    if sec(tilt >= 5) < 20:
        gaps.append("倾角 ≥5° 的帧不足 20 s —— 姿态变化没覆盖到")
    for g_ in gaps:
        print(f"  ⚠ {g_}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, nargs="+", required=True,
                    help="记录前缀（不含 _joint500.bin），可以给多个")
    ap.add_argument("--out", type=Path, default=spec.TRACTION_LOG_DIR / "real_finetune")
    ap.add_argument("--steady-s", type=float, default=0.6,
                    help="指令稳住多久算稳态。这些帧的速度标签给满权重")
    ap.add_argument("--tau-ms", type=float, default=400.0,
                    help="指令低通的时间常数，近似策略的速度阶跃响应（真机实测 ~460 ms）")
    ap.add_argument("--transient-weight", type=float, default=0.5,
                    help="非稳态帧速度标签的权重折扣")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--skip-json", type=Path, default=None,
                    help="排除窗 json：{录制名: [[t0,t1],…]}，t 是 policy 帧时间（帧号/50）。"
                         "录制里有碰撞/吊拉/被拖拽的段时必须给 —— 那些帧有外力，"
                         "拿去当'力=0'的标签是在教估计器低估")
    args = ap.parse_args()
    skips = {}
    if args.skip_json:
        skips = {k: v for k, v in json.load(open(args.skip_json)).items()
                 if not k.startswith("_")}

    all_hist, all_cmd, all_lp, all_steady, all_src = [], [], [], [], []
    for r, rec in enumerate(args.rec):
        frames, cmd, usable = build_frames(rec)
        L = spec.HISTORY_LENGTH
        idx = np.arange(L)[None, :] + np.arange(len(frames) - L + 1)[:, None]
        skip = int(args.warmup * spec.POLICY_HZ)
        idx = idx[skip:]
        aligned = cmd[L - 1 + skip:]
        keep = usable[idx].all(axis=1)
        idx, aligned = idx[keep], aligned[keep]

        ranges = skips.get(rec.name, [])
        if ranges:
            tw = idx[:, -1] / spec.POLICY_HZ          # 窗口末帧（=标签帧）的时间
            bad = np.zeros(len(tw), dtype=bool)
            for lo, hi in ranges:
                bad |= (tw >= lo - 0.5) & (tw <= hi + 0.5)   # ±0.5s：盖住整个历史窗
            idx, aligned = idx[~bad], aligned[~bad]
            print(f"[{rec.name}] 排除窗丢掉 {int(bad.sum()):,} 窗"
                  f"（{bad.sum()/spec.POLICY_HZ:.0f} s，{len(ranges)} 段）")

        st = steady_mask(aligned, args.steady_s)
        all_hist.append(frames[idx])
        all_cmd.append(aligned)
        all_lp.append(lowpass(aligned, args.tau_ms))
        all_steady.append(st)
        all_src.append(np.full(len(idx), r, dtype=np.int16))
        print(f"[{rec.name}] {len(idx):,} 窗 = {len(idx)/spec.POLICY_HZ:.0f} s，"
              f"其中指令稳态 {st.sum():,}（{100*st.mean():.0f}%）")

    hist = np.concatenate(all_hist).astype(np.float32)
    cmd = np.concatenate(all_cmd).astype(np.float32)
    lp = np.concatenate(all_lp).astype(np.float32)
    steady = np.concatenate(all_steady)
    src = np.concatenate(all_src)

    # 速度标签：机身系 (vx, vy, vz)。vz 没有标签，权重给 0。
    vel = np.zeros((len(cmd), 3), dtype=np.float32)
    vel[:, 0], vel[:, 1] = lp[:, 0], lp[:, 1]
    w = np.where(steady, 1.0, args.transient_weight).astype(np.float32)
    vel_weight = np.stack([w, w, np.zeros_like(w)], axis=1)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "real.npz", history=hist, cmd=cmd,
                        vel_label=vel, vel_weight=vel_weight, steady=steady, source=src,
                        sources=np.array([str(r) for r in args.rec]))
    coverage_report(hist, cmd, src, [str(r) for r in args.rec])
    print(f"\n[traction] {len(hist):,} 个窗口 → {args.out / 'real.npz'}")
    print(f"           力标签：全部 {len(hist):,} 帧，真值 0 N（采集时没人碰）")
    print(f"           速度标签：全部 {len(hist):,} 帧（指令低通 τ={args.tau_ms:g} ms），"
          f"稳态 {int(steady.sum()):,} 帧满权重，其余 ×{args.transient_weight:g}")
    moving = np.abs(lp[:, 0]) > 0.3
    print(f"           其中走起来的 {int(moving.sum()):,} 帧 = {moving.sum()/spec.POLICY_HZ:.0f} s"
          f" —— 掰正速度头主要靠这些")


if __name__ == "__main__":
    main()
