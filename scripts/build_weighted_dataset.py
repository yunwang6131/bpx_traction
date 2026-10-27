#!/usr/bin/env python3
"""配重法：用已知质量的吊重给真机数据打**有力时的**力标签。

    python scripts/build_weighted_dataset.py \
        --rec logs/real_v6/rec_20260419_101530 \
        --pull front --mass-kg 2.0 --window 12 42 \
        --rec logs/real_v6/rec_20260419_101530 \
        --pull left  --mass-kg 2.0 --window 55 85 \
        --out logs/traction/real_weighted

## 为什么需要这个

真机上现在只有**单边**标签：无外力录制能给"力 = 0"，但给不出"有力时该是多少"。
只靠单边监督，网络的最优解是"永远输出零" —— 这不是假想的风险，装错结构那个模型
（仿真 19.25 N，基本不响应）在无外力数据上看着误报最低，就是这个样子。

吊重把力的**大小**变成已知量：绳子绕过定滑轮，另一端挂 m 公斤，张力就是 m·g。
不需要拉力计，也不需要人手保持恒定。

## 物理设置

    机器人 ──绳──→ [定滑轮，桌沿/支架] ──┐
                                        │
                                       配重 m kg

* 绳子从机器人到滑轮这一段必须**水平**，否则张力有竖直分量，和 `--pull` 的水平假设不符。
* 绳子要长（≥3 m）：机器人被拉动 0.3 m 时方向只偏 6°，可以忽略。短绳不行。
* 滑轮要用**带轴承的**。羊眼圈/挂钩的摩擦能吃掉 15–30% 的张力，而且方向相关，
  标定不出来。用 `--efficiency` 填实测效率，见下面怎么量。

## 怎么量滑轮效率

挂重之前，在绳子和机器人之间串一个数显行李秤，挂上配重静置，读数除以 m·g 就是 η。
各个方向各量一次（摩擦是方向相关的）。不量的话默认 1.0，**标签会系统性偏高**。

## 标签怎么算

`--pull` 说的是**在 t0 那一刻，绳子朝机身的哪个方向拉**（前/后/左/右）。脚本据此：

1. 从窗口第一帧的四元数算出这个机身方向在**世界系**里指向哪（绳子在世界系里固定）；
2. 逐帧用当前四元数把这个世界方向转回机身系 —— 机器人被拉歪、转头都能跟上；
3. 乘上 m·g·η 得到机身系的力向量。

旋转约定由录制自带的 `projected_gravity_b` 当场验证，对不上直接报错。
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

G = 9.80665
#: `--pull` 的取值 → t0 时刻绳子在**机身系**里的拉拽方向（单位向量）。
PULL_DIRS = {"front": (1.0, 0.0, 0.0), "back": (-1.0, 0.0, 0.0),
             "left": (0.0, 1.0, 0.0), "right": (0.0, -1.0, 0.0)}
#: 窗口内允许的最大机身转动。超过说明机器人被拖着转了一大圈，绳子的世界方向不再固定。
MAX_YAW_DRIFT_DEG = 25.0
#: 窗口内允许的最大关节速度中位数。配重法要求机器人基本站住 —— 它一走，
#: 配重就在升降，张力不再等于 m·g。
MAX_STANDING_DQ = 1.5


def quat_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """(N,4) wxyz → (N,3,3) 的 R(body→world)。"""
    q = q_wxyz / np.linalg.norm(q_wxyz, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def check_rotation(frames: np.ndarray, R: np.ndarray) -> float:
    """用录制自带的 `projected_gravity_b` 验旋转矩阵，返回最大偏差。

    `projected_gravity_b` 就是 ``Rᵀ · [0, 0, −1]``（板子 `projected_gravity_xyzw`
    的定义，真机实测站立时是 [-0.0075, 0.0047, -0.9993]）。两边对不上就说明四元数
    顺序或旋转方向搞反了 —— 那样算出来的力方向会整个错掉，而且不会报错。
    """
    want = frames[:, slice(*spec.term_slice("projected_gravity_b"))]
    got = -R.transpose(0, 2, 1) @ np.array([0.0, 0.0, 1.0])
    return float(np.abs(got - want).max())


def one_segment(rec: Path, pull: str, mass_kg: float, window, efficiency: float):
    frames, cmd, usable = build_frames(rec)
    L = spec.HISTORY_LENGTH
    idx = np.arange(L)[None, :] + np.arange(len(frames) - L + 1)[:, None]
    keep = usable[idx].all(axis=1)
    idx = idx[keep]
    # 窗口是**秒**，相对录制里策略段的开头
    t = np.arange(len(frames))[L - 1:][keep] / spec.POLICY_HZ
    sel = (t >= window[0]) & (t < window[1])
    if sel.sum() < L:
        raise SystemExit(f"{rec.name} 的窗口 {window} 里只有 {int(sel.sum())} 帧，太少")
    idx, t = idx[sel], t[sel]
    last = frames[idx[:, -1]]

    R = quat_to_matrix(last[:, slice(*spec.term_slice("quat_wb"))].astype(np.float64))
    err = check_rotation(last, R)
    if err > 0.05:
        raise SystemExit(f"{rec.name}: 四元数旋转和 projected_gravity 对不上（最大差 {err:.3f}）")

    # 绳子在世界系里的方向：t0 那一刻的机身方向转到世界系，拍平到水平面
    world = R[0] @ np.asarray(PULL_DIRS[pull], dtype=np.float64)
    world[2] = 0.0
    n = np.linalg.norm(world)
    if n < 1e-6:
        raise SystemExit(f"{rec.name}: t0 姿态太竖，拉拽方向投影到水平面后退化")
    world /= n

    yaw_drift = np.degrees(np.arccos(np.clip(
        (R[:, :, 0] @ R[0, :, 0]) / 1.0, -1.0, 1.0)))
    dq = np.linalg.norm(last[:, slice(*spec.term_slice("joint_vel"))], axis=1)
    warn = []
    if yaw_drift.max() > MAX_YAW_DRIFT_DEG:
        warn.append(f"机身转了 {yaw_drift.max():.0f}°（>{MAX_YAW_DRIFT_DEG:.0f}°），"
                    "绳子的世界方向不再固定，标签方向会偏")
    if np.median(dq) > MAX_STANDING_DQ:
        warn.append(f"‖dq‖ 中位数 {np.median(dq):.2f}（>{MAX_STANDING_DQ}），"
                    "机器人在走 —— 配重在升降，张力不等于 m·g")

    body_dir = np.einsum("nji,j->ni", R, world)      # Rᵀ · world
    force = (mass_kg * G * efficiency) * body_dir
    return {"history": frames[idx], "force": force.astype(np.float32),
            "t": t, "rotation_check": err, "yaw_drift_deg": float(yaw_drift.max()),
            "dq_median": float(np.median(dq)), "warnings": warn,
            "fz_frac": float(np.abs(body_dir[:, 2]).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, action="append", required=True,
                    help="记录前缀；每段配重各给一次，和 --pull/--mass-kg/--window 一一对应")
    ap.add_argument("--pull", action="append", required=True, choices=list(PULL_DIRS),
                    help="t0 时绳子朝机身哪个方向拉")
    ap.add_argument("--mass-kg", type=float, action="append", required=True)
    ap.add_argument("--window", type=float, nargs=2, action="append", required=True,
                    metavar=("开始秒", "结束秒"), help="这段配重稳定悬挂的时间区间")
    ap.add_argument("--efficiency", type=float, default=1.0,
                    help="滑轮效率 η，张力 = m·g·η。**务必实测**（见模块文档），"
                         "默认 1.0 会让标签系统性偏高")
    ap.add_argument("--out", type=Path, default=spec.TRACTION_LOG_DIR / "real_weighted")
    args = ap.parse_args()

    n = len(args.rec)
    if not (len(args.pull) == len(args.mass_kg) == len(args.window) == n):
        raise SystemExit(f"--rec/--pull/--mass-kg/--window 的个数要一样，"
                         f"现在是 {n}/{len(args.pull)}/{len(args.mass_kg)}/{len(args.window)}")
    if args.efficiency == 1.0:
        print("⚠ --efficiency 没给，按 1.0 算。带轴承滑轮约 0.95–0.98，"
              "羊眼圈/挂钩能低到 0.7 —— 不实测的话标签会系统性偏高。\n")

    hist, force, src, meta = [], [], [], []
    for k in range(n):
        s = one_segment(args.rec[k], args.pull[k], args.mass_kg[k],
                        args.window[k], args.efficiency)
        mag = args.mass_kg[k] * G * args.efficiency
        print(f"[{args.rec[k].name}] {args.pull[k]:<5} {args.mass_kg[k]:.1f} kg → "
              f"{mag:5.1f} N   {len(s['history']):,} 窗 = "
              f"{len(s['history'])/spec.POLICY_HZ:.0f} s")
        print(f"    旋转自检 {s['rotation_check']:.4f}   机身转动 {s['yaw_drift_deg']:.1f}°   "
              f"‖dq‖中位 {s['dq_median']:.2f}   |Fz|/|F| 均值 {s['fz_frac']:.3f}")
        for w in s["warnings"]:
            print(f"    ⚠ {w}")
        hist.append(s["history"])
        force.append(s["force"])
        src.append(np.full(len(s["history"]), k, dtype=np.int16))
        meta.append({"rec": str(args.rec[k]), "pull": args.pull[k],
                     "mass_kg": args.mass_kg[k], "window_s": list(args.window[k]),
                     "force_n": mag, "frames": int(len(s["history"])),
                     "rotation_check": s["rotation_check"],
                     "yaw_drift_deg": s["yaw_drift_deg"], "dq_median": s["dq_median"],
                     "warnings": s["warnings"]})

    H = np.concatenate(hist).astype(np.float32)
    F = np.concatenate(force).astype(np.float32)
    S = np.concatenate(src)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "weighted.npz", history=H, force_label=F, source=S)
    (args.out / "manifest.json").write_text(
        json.dumps({"efficiency": args.efficiency, "segments": meta}, indent=2,
                   ensure_ascii=False) + "\n")

    mags = np.linalg.norm(F[:, :2], axis=1)
    print(f"\n[traction] {len(H):,} 个带力标签的真机窗口 → {args.out / 'weighted.npz'}")
    print(f"           |F_xy| 覆盖 {mags.min():.1f} – {mags.max():.1f} N，"
          f"设计范围 {spec.FORCE_DESIGN_RANGE_N[0]:.0f}–{spec.FORCE_DESIGN_RANGE_N[1]:.0f} N")
    covered = {k: int(((mags >= lo) & (mags < hi)).sum())
               for k, (lo, hi) in {"0–5": (0, 5), "5–15": (5, 15),
                                   "15–25": (15, 25), "25–30": (25, 30),
                                   ">30": (30, 1e9)}.items()}
    print("           分档：" + "  ".join(f"{k} {v:,}" for k, v in covered.items()))
    missing = [d for d in PULL_DIRS if d not in args.pull]
    if missing:
        print(f"           ⚠ 没采的方向：{', '.join(missing)} —— 方向覆盖不全，"
              "估计器会在缺的方向上系统性偏")


if __name__ == "__main__":
    main()
