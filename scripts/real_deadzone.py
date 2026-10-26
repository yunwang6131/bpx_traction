#!/usr/bin/env python3
"""把真机 v6 记录组装成 61 维帧，跑力估计器，量出走路时的**误报力** → 死区。

    python scripts/real_deadzone.py --rec logs/real_v6/rec_20260418_191052 \
                                    [--estimator logs/traction/estimator_residual]

真机采集全程没有外力（没人碰它），所以估计器吐出来的任何非零力都是误报。
死区必须跨过这个分布的上尾，否则机器人会自己走起来。

帧的来源：
  policy50 的 obs00..obs44 是板子自己拼的 45 维策略观测，顺序由 bpx_rma 契约固定，
  里面有 joint_pos_rel / joint_vel / omega / projected_gravity / last_action。
  剩下的 acc(3) / quat(4) / tau(12) 从 joint500 按时间戳最近邻取。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_traction import spec  # noqa: E402
from bpx_traction.model import ForceTCN  # noqa: E402

JOINTS = [f"{leg}_{j}" for leg in ("fl", "fr", "hl", "hr") for j in ("roll", "pitch", "knee")]
#: 45 维策略观测里各段的位置，顺序来自 deploy.yaml 的 observations 段。
OBS = {"joint_pos_rel": (0, 12), "joint_vel": (12, 24), "imu_gyro_b": (24, 27),
       "projected_gravity_b": (27, 30), "last_action": (30, 42), "cmd": (42, 45)}


def read(stem: Path, tag: str):
    cols = json.loads(stem.with_name(stem.name + f"_{tag}.json").read_text())["columns"]
    raw = np.fromfile(stem.with_name(stem.name + f"_{tag}.bin"), dtype="<f8")
    if raw.size % len(cols):
        raise SystemExit(f"{stem.name}_{tag}: {raw.size} 个数除不尽 {len(cols)} 列")
    return raw.reshape(-1, len(cols)), {c: k for k, c in enumerate(cols)}


#: 500 Hz 快照和 50 Hz 推理之间允许的最大时间偏差（两个 500 Hz tick）。
MAX_SKEW_S = 0.004
#: policy50 行的 ``t`` 相对它所用观测的滞后（实测 −2 ms，即一个 500 Hz tick）。
OBS_LAG_S = -0.002


def build_frames(stem: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (frames (N,61), cmd (N,3), usable (N,))。

    ``usable[t]`` 为假的帧不能作为历史窗的一部分：要么和 500 Hz 快照对不齐，
    要么跨过了策略重启（session 变化 / 时间空洞）。
    """
    pol, pi = read(stem, "policy50")
    jnt, ji = read(stem, "joint500")

    # joint500 里只有 JOINT/POLICY 才有行；按时间戳给每个推理配一行最近的 500 Hz 快照。
    tj, tp = jnt[:, ji["t"]], pol[:, pi["t"]]
    # policy50 的 t 是推理**之后**写的，obs 却是推理**之前**采的；实测偏移一个 500 Hz tick
    # （用 obs 里的 joint_pos/joint_vel/gyro 和 joint500 对扫出来的，见 check_real_frame.py）。
    tq = tp + OBS_LAG_S
    k = np.searchsorted(tj, tq).clip(1, len(tj) - 1)
    k = np.where(np.abs(tj[k] - tq) < np.abs(tj[k - 1] - tq), k, k - 1)
    skew = np.abs(tj[k] - tq)

    # 策略重启会让 session 跳变、step 回退、时间出现空洞；历史窗不能跨过去。
    gap = np.r_[False, np.diff(tp) > 3 * spec.SAMPLE_PERIOD_S]
    restart = np.r_[False, np.diff(pol[:, pi["session"]]) != 0] | gap
    usable = (skew <= MAX_SKEW_S) & ~restart
    print(f"[对齐] 偏差 p99 {np.percentile(skew, 99)*1000:.2f} ms；"
          f"丢弃 对不齐 {int((skew > MAX_SKEW_S).sum())} 帧 + 重启断点 {int(restart.sum())} 处")

    obs = pol[:, [pi["obs%02d" % i] for i in range(45)]]
    quat_xyzw = jnt[np.ix_(k, [ji[c] for c in ("quat_x", "quat_y", "quat_z", "quat_w")])]

    frames = np.empty((len(pol), spec.FRAME_SIZE), dtype=np.float32)
    put = lambda name, v: frames.__setitem__((slice(None), slice(*spec.term_slice(name))), v)
    put("imu_acc_b", jnt[np.ix_(k, [ji[c] for c in ("acc_x", "acc_y", "acc_z")])])
    put("joint_torque", jnt[np.ix_(k, [ji["tau_" + n] for n in JOINTS])])
    put("quat_wb", quat_xyzw[:, [3, 0, 1, 2]])          # 板子是 xyzw，契约要 wxyz
    for name, (a, b) in OBS.items():
        if name != "cmd":
            put(name, obs[:, a:b])
    return frames, obs[:, slice(*OBS["cmd"])], usable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True, help="记录文件前缀，不含 _joint500.bin")
    ap.add_argument("--estimator", type=Path, default=None)
    ap.add_argument("--warmup", type=float, default=2.0, help="丢掉开头多少秒（历史没填满）")
    ap.add_argument("--hold", type=float, default=0.20, help="映射层的 engage_hold_s 去抖时长")
    args = ap.parse_args()

    estimator = args.estimator or spec.DEFAULT_ESTIMATOR_DIR
    frames, cmd, usable = build_frames(args.rec)
    print(f"[帧] {len(frames)} 次推理 = {len(frames)/50:.1f} s，{spec.FRAME_SIZE} 维")

    acc_norm = np.linalg.norm(frames[:, slice(*spec.term_slice("imu_acc_b"))], axis=1).mean()
    if not 8.5 < acc_norm < 11.0:
        raise SystemExit(f"加速度模长均值 {acc_norm:.2f}，不像含重力的比力 —— 契约对不上")
    print(f"[检查] 加速度模长均值 {acc_norm:.2f} m/s² ✓ 含重力")

    ck = torch.load(estimator / "estimator.pt", map_location="cpu", weights_only=False)
    model = ForceTCN(input_dim=ck["input_dim"], channels=ck["channels"],
                     blocks=ck["blocks"], dilations=tuple(ck["dilations"]))
    model.load_state_dict(ck["model"])
    model.eval()
    nz = np.load(estimator / "normalization.npz")
    mean, std = nz["mean"].astype(np.float32), nz["std"].astype(np.float32)
    print(f"[模型] {estimator.name}  仿真验证集 xy RMSE {ck['score_xy_rmse_n']:.3f} N")

    # 滑窗成历史：第 t 帧用 [t-24, t]，最旧在前。整窗都 usable 才要。
    L = spec.HISTORY_LENGTH
    idx = np.arange(L)[None, :] + np.arange(len(frames) - L + 1)[:, None]
    skip = int(args.warmup * 50)
    idx, cmd = idx[skip:], cmd[L - 1 + skip:]
    whole = usable[idx].all(axis=1)
    idx, cmd = idx[whole], cmd[whole]
    print(f"[窗口] {len(idx)} 个完整历史窗（丢掉 {int((~whole).sum())} 个跨断点/对不齐的）")

    fb, _ = spec.output_slice("force_b")
    preds, confs = [], []
    with torch.no_grad():
        for a in range(0, len(idx), 2048):
            x = (frames[idx[a:a + 2048]] - mean) / std
            v, c = model(torch.from_numpy(x))
            preds.append(v.numpy()[:, fb:fb + 2])
            confs.append(c.numpy()[:, spec.CONFIDENCE_TERMS.index("force")])
    fxy = np.linalg.norm(np.concatenate(preds), axis=1)
    conf = np.concatenate(confs)
    print(f"[推理] {len(fxy)} 个样本（丢掉开头 {args.warmup:g}s 热身）\n")

    q = lambda v, p: np.percentile(v, p) if len(v) else float("nan")
    print("走路时的**误报力** |F_xy|（全程无外力，任何非零都是误报）")
    print(f"  {'工况':<16}{'样本':>8}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    speed = np.abs(cmd[:, 0])
    lateral = np.abs(cmd[:, 1])
    yaw = np.abs(cmd[:, 2])
    moving = (speed > 0.05) | (lateral > 0.05) | (yaw > 0.05)
    buckets = [
        ("站立（零指令）", ~moving),
        ("前进 0.05–0.6", (speed > 0.05) & (speed <= 0.6)),
        ("前进 0.6–1.2", (speed > 0.6) & (speed <= 1.2)),
        ("前进 >1.2", speed > 1.2),
        ("侧移", lateral > 0.05),
        ("转向", yaw > 0.05),
        ("── 全部运动", moving),
    ]
    for name, m in buckets:
        v = fxy[m]
        if not len(v):
            continue
        print(f"  {name:<16}{len(v):>8}{q(v,50):>8.2f}{q(v,90):>8.2f}"
              f"{q(v,95):>8.2f}{q(v,99):>8.2f}{v.max():>8.2f}")

    print(f"\n置信度闸门：force conf p50 {q(conf,50):.3f}  p95 {q(conf,95):.3f}")
    gated = fxy[conf >= 0.55]
    print(f"  conf≥0.55 的样本 {len(gated)}/{len(fxy)} ({100*len(gated)/len(fxy):.1f}%)"
          + (f"，其 |F_xy| p99 {q(gated,99):.2f}  max {gated.max():.2f} N" if len(gated) else ""))

    # 真正决定死区的不是瞬时 p99，而是**连续超阈值 engage_hold_s 才起步**之后还剩多少误触发。
    hold = int(round(args.hold * 50))
    print(f"\n加上 {args.hold:g}s 持续去抖（连续 {hold} 帧超阈值才算一次误起步）：")
    print(f"  {'阈值N':>7}{'瞬时超阈':>11}{'误起步次数':>12}{'每分钟':>9}{'最长一次s':>11}")
    for thr in (10, 15, 20, 25, 30, 35, 40, 45, 50):
        pad = np.r_[False, fxy > thr, False]
        starts = np.flatnonzero(~pad[:-1] & pad[1:])
        runs = np.flatnonzero(pad[:-1] & ~pad[1:]) - starts
        fires = int((runs >= hold).sum())
        print(f"  {thr:>7}{100*(fxy > thr).mean():>10.1f}%{fires:>12}"
              f"{fires/(len(fxy)/50/60):>9.2f}{(runs.max()/50 if len(runs) else 0):>11.2f}")

    mv = fxy[moving]
    print("\n" + "=" * 62)
    print(f"运动时误报力 p99 = {q(mv,99):.2f} N   max = {mv.max():.2f} N")
    print(f"死区建议：off(松开) ≥ {q(mv,99):.1f} N   on(起步) ≥ {mv.max():.1f} N")
    print("=" * 62)


if __name__ == "__main__":
    main()
