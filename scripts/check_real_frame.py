#!/usr/bin/env python3
"""核对真机组装出来的 61 维帧和训练分布对不对得上。

估计器在真机上误报很大时，先用这个分清是**模型不迁移**还是**帧拼错了**：
拼错会表现为某几列的 z 分数离谱，模型不迁移则是各列都在范围内但输出不对。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_traction import spec  # noqa: E402
from real_deadzone import JOINTS, OBS, build_frames, read  # noqa: E402

Q0 = np.array([0.0, 0.7, -1.4] * 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", type=Path, required=True)
    ap.add_argument("--estimator", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=Path("logs/traction/dataset_fixed"))
    args = ap.parse_args()

    estimator = args.estimator or spec.DEFAULT_ESTIMATOR_DIR
    frames, _cmd, usable = build_frames(args.rec)
    frames = frames[usable]
    nz = np.load(estimator / "normalization.npz")
    mean, std = nz["mean"], nz["std"]

    print("\n【一】板子自己拼的 45 维观测 vs 我从 joint500 独立算的")
    pol, pi = read(args.rec, "policy50")
    jnt, ji = read(args.rec, "joint500")
    tj, tp = jnt[:, ji["t"]], pol[:, pi["t"]]
    k = np.searchsorted(tj, tp).clip(1, len(tj) - 1)
    k = np.where(np.abs(tj[k] - tp) < np.abs(tj[k - 1] - tp), k, k - 1)
    obs = pol[:, [pi["obs%02d" % i] for i in range(45)]]
    ok = np.abs(tj[k] - tp) <= 0.004
    for name, src in (("joint_pos_rel", jnt[np.ix_(k, [ji["q_" + n] for n in JOINTS])] - Q0),
                      ("joint_vel", jnt[np.ix_(k, [ji["dq_" + n] for n in JOINTS])]),
                      ("imu_gyro_b", jnt[np.ix_(k, [ji[c] for c in ("omega_x", "omega_y", "omega_z")])])):
        a, b = OBS[name]
        err = np.abs(obs[ok, a:b] - src[ok]).max()
        print(f"  {name:<16} 最大差 {err:.5f}  {'✓ 一致' if err < 0.05 else '✗ 对不上！'}")

    print("\n【二】各列的 z 分数（真机帧按训练集均值/方差归一化后）")
    z = (frames - mean) / std
    print(f"  {'段':<20}{'|z| p50':>10}{'|z| p99':>10}{'|z| max':>10}   判定")
    bad = []
    for (name, _dim), (a, b) in zip(spec.TERMS, spec.term_slices()):
        zz = np.abs(z[:, a:b])
        p50, p99, mx = np.percentile(zz, 50), np.percentile(zz, 99), zz.max()
        flag = "✗ 严重超出" if p99 > 6 else ("⚠ 偏大" if p99 > 3 else "✓")
        if p99 > 3:
            bad.append(name)
        print(f"  {name:<20}{p50:>10.2f}{p99:>10.2f}{mx:>10.2f}   {flag}")

    print("\n【三】逐列对照：真机 vs 训练集（原始量纲）")
    tr = np.load(args.dataset / "validation.npz")["history"].reshape(-1, spec.FRAME_SIZE)
    names = spec.column_names() if hasattr(spec, "column_names") else None
    print(f"  {'列':<22}{'真机均值':>10}{'训练均值':>10}{'真机std':>10}{'训练std':>10}")
    for (name, _dim), (a, b) in zip(spec.TERMS, spec.term_slices()):
        if name not in bad:
            continue
        for c in range(a, b):
            lbl = names[c] if names else f"{name}[{c-a}]"
            print(f"  {lbl:<22}{frames[:, c].mean():>10.3f}{tr[:, c].mean():>10.3f}"
                  f"{frames[:, c].std():>10.3f}{tr[:, c].std():>10.3f}")

    print("\n【四】训练集里有没有静止站立这种工况")
    sp = np.linalg.norm(tr[:, slice(*spec.term_slice("joint_vel"))], axis=1)
    rs = np.linalg.norm(frames[:, slice(*spec.term_slice("joint_vel"))], axis=1)
    print(f"  ‖joint_vel‖  训练 p1 {np.percentile(sp,1):.2f} p50 {np.percentile(sp,50):.2f} "
          f"p99 {np.percentile(sp,99):.2f}")
    print(f"               真机 p1 {np.percentile(rs,1):.2f} p50 {np.percentile(rs,50):.2f} "
          f"p99 {np.percentile(rs,99):.2f}")


if __name__ == "__main__":
    main()
