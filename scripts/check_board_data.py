#!/usr/bin/env python3
"""核对 v6 采回来的真机数据，回答六个悬着的问题。

    python scripts/check_board_data.py --root logs/real_v6

不需要 Isaac Sim、不需要 GPU。读 `rec_*_joint500.bin` 和旁边的 `.json` 列名。

每一条都是**仿真里回答不了、只能靠真机数据定**的：

1. `snap_us` 的分布 —— 加了两个 getter 之后 500 Hz 还剩多少余量
2. 加速度 z 的均值 —— 确认是**比力**（含重力）不是运动学加速度。这条搞反，
   仿真训出来的估计器一上真机就废，而且在仿真里永远测不出来
3. 四元数顺序 —— 手册 6.3 说 (w,x,y,z)，板子的 `projected_gravity_xyzw()` 按 xyzw 解，
   8.5 的高频版没标。**这条没有任何文档支撑**，门 2 也不覆盖它
4. 静止段的噪声本底 —— 仿真采集要注入多大噪声
5. 走路段的力矩波动 —— **死区该设多少**（仿真给的是 5.1 N）
6. 力矩读数的量级和符号 —— 和站立时的重力力矩对不对得上
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_rma import spec as rma_spec  # noqa: E402


def load(path: Path):
    """读一份 rec_*.bin + 同名 .json 的列名。

    行数对不上列名就报错退出 —— 多半是 .json 和 .bin 来自不同版本的记录器，
    硬 reshape 会把整份数据错位，而错位之后每一列看上去都"有数"，查不出来。
    """
    meta = json.loads(path.with_suffix(".json").read_text())
    cols = meta["columns"]
    raw = np.fromfile(path, dtype=meta.get("dtype", "<f8"))
    if raw.size % len(cols):
        raise ValueError(
            f"{path.name}：{raw.size} 个数除不尽 {len(cols)} 列。"
            f"\n  .json 和 .bin 对不上（不同版本的记录器？），不要硬 reshape —— "
            f"错位之后每列都像有数，但全是错的。"
        )
    return raw.reshape(-1, len(cols)), {name: i for i, name in enumerate(cols)}


def second_diff_sigma(x: np.ndarray) -> np.ndarray:
    """白噪声标准差估计：var(x[n] − 2x[n−1] + x[n−2]) = 6σ²。

    和 `scripts/measure_sensor_noise.py` 同一个办法 —— 真实动态在 20 Hz 以下，
    噪声是宽带的，二阶差分能把两者分开。
    """
    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
    return d2.std(axis=0) / np.sqrt(6.0)


def report_timing(data, idx) -> list[str]:
    if "snap_us" not in idx:
        return ["⚠️ 没有 snap_us 列 —— 这份不是 v6 采的"]
    us = data[:, idx["snap_us"]]
    us = us[np.isfinite(us)]
    budget = 1e6 / rma_spec.JOINT_HZ          # 500 Hz → 2000 µs
    lines = [
        f"快照耗时   p50 {np.median(us):.0f}   p95 {np.percentile(us, 95):.0f}   "
        f"p99 {np.percentile(us, 99):.0f}   max {us.max():.0f} µs",
        f"           500 Hz 预算 {budget:.0f} µs/步，p99 占 {np.percentile(us, 99) / budget:.0%}",
    ]
    if np.percentile(us, 99) > 0.5 * budget:
        lines.append("           ⚠️ p99 超过预算一半，加上策略推理可能不够用")
    return lines


def report_accel(data, idx) -> list[str]:
    keys = [f"acc_{a}" for a in "xyz"]
    if not all(k in idx for k in keys):
        return ["⚠️ 没有 acc 列 —— 这份不是 v6 采的"]
    acc = data[:, [idx[k] for k in keys]]
    mean = acc.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    lines = [
        f"加速度均值 {np.round(mean, 2)}   模长 {norm:.2f} m/s²",
        f"           p99 {np.round(np.percentile(np.abs(acc), 99, axis=0), 1)}   "
        f"max {np.round(np.abs(acc).max(axis=0), 1)}",
    ]
    if abs(norm - 9.81) < 2.0:
        lines.append("           ✓ 模长 ≈9.81，是**比力**（含重力），和仿真的 "
                     "ImuCfg(gravity_bias=(0,0,9.81)) 一致")
    elif norm < 3.0:
        lines.append("           ✗ 模长接近 0 —— 这是**运动学加速度**，不含重力。"
                     "仿真那边必须跟着把 gravity_bias 设成 (0,0,0)，否则 sim2real 直接废")
    else:
        lines.append(f"           ⚠️ 模长 {norm:.2f} 既不是 0 也不是 9.81，查一下单位")
    return lines


def report_quaternion(data, idx) -> list[str]:
    keys = ["quat_x", "quat_y", "quat_z", "quat_w"]
    if not all(k in idx for k in keys):
        return ["⚠️ 没有四元数列"]
    q = data[:, [idx[k] for k in keys]]           # 记录器按 x,y,z,w 存
    norm = np.linalg.norm(q, axis=1)
    lines = [f"四元数     模长 均值 {norm.mean():.4f}（该 ≈1）"]

    # 判别顺序**不能**看"哪一位最大"。平地上机器人会转 yaw，而纯 yaw 旋转下实部和
    # z 分量都是大的（w=cos(yaw/2)、z=sin(yaw/2)），两者分不开。
    #
    # 正确的判别是**哪两位接近零**：平地行走 roll/pitch ≈ 0，所以 x、y 两个分量恒小。
    #   xyzw → 小的在第 1、2 位
    #   wxyz → 小的在第 2、3 位
    mean_abs = np.abs(q).mean(axis=0)
    small = set(int(i) for i in np.argsort(mean_abs)[:2])
    lines.append(f"           各位 |均值| {np.round(mean_abs, 3)}  "
                 f"（接近零的两位 = roll/pitch 分量 x,y）")
    if small == {0, 1}:
        lines.append("           ✓ x,y 在第 1、2 位 → SDK 返回 **xyzw**，"
                     "板子的 projected_gravity_xyzw() 解法正确")
    elif small == {1, 2}:
        lines.append("           ⚠️ x,y 在第 2、3 位 → SDK 返回 **wxyz**，"
                     "而板子按 xyzw 解 —— 投影重力一直是错的，复核板子代码")
    else:
        lines.append(f"           ⚠️ 接近零的是第 {sorted(i + 1 for i in small)} 位，"
                     "判别不了 —— 这段数据里机身倾角太大？换静止段再看")
    lines.append("           （估计器契约 spec.QUAT_ORDER 是 wxyz，板端适配器要按实测结果重排）")
    return lines


def report_torque(data, idx, standing: bool) -> list[str]:
    names = [f"{leg}_{jt}" for leg in ("fl", "fr", "hl", "hr")
             for jt in ("roll", "pitch", "knee")]
    keys = [f"tau_{n}" for n in names]
    if not all(k in idx for k in keys):
        return ["⚠️ 没有 tau 列 —— 这份不是 v6 采的"]
    tau = data[:, [idx[k] for k in keys]]
    lines = [
        f"力矩       |τ| p50 {np.percentile(np.abs(tau), 50):.2f}   "
        f"p99 {np.percentile(np.abs(tau), 99):.2f}   max {np.abs(tau).max():.2f} N·m",
    ]
    if standing:
        mean = tau.mean(axis=0)
        lines.append("           站立时各关节均值（和 URDF 的重力力矩对比，验符号和标度）：")
        for group, offset in (("roll ", 0), ("pitch", 1), ("knee ", 2)):
            vals = [mean[i] for i in range(offset, 12, 3)]
            lines.append(f"             {group} {np.round(vals, 2)}")
        knee = np.abs([mean[i] for i in range(2, 12, 3)]).mean()
        if knee < 0.5:
            lines.append("           ⚠️ 膝关节力矩均值接近 0 —— 站立时膝要撑体重，"
                         "不该这么小。读数可能是前馈指令而不是测量值")
    return lines


def report_body_force(data, idx) -> list[str]:
    """走路时的机身力波动 —— 死区的依据。

    没有力传感器，所以用**关节力矩的波动**当代理量：静止时的力矩是重力的静态分布，
    走路时的波动量反映了步态产生的额外内力。估计器会把这部分当成外力的那一截，
    就是死区必须跨过去的量。
    """
    names = [f"{leg}_{jt}" for leg in ("fl", "fr", "hl", "hr")
             for jt in ("roll", "pitch", "knee")]
    keys = [f"tau_{n}" for n in names]
    if not all(k in idx for k in keys):
        return []
    tau = data[:, [idx[k] for k in keys]]
    fluctuation = tau.std(axis=0)
    return [
        f"力矩波动   逐关节 std p50 {np.median(fluctuation):.2f}   "
        f"max {fluctuation.max():.2f} N·m",
        "           这是步态自己产生的内力 —— 估计器会把其中一部分当成外力，"
        "死区必须跨过去",
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="核对 v6 真机数据")
    ap.add_argument("--root", type=str, default="logs/real_v6")
    ap.add_argument("--standing", type=str, nargs="*", default=["01", "stand"],
                    help="文件名里含这些字样的当成静止段（力矩标定用）")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = PROJECT_DIR / root
    files = sorted(root.glob("rec_*_joint500.bin"))
    if not files:
        print(f"{root} 下没有 rec_*_joint500.bin")
        print("先按 bpx_traction/board/README.md 采数据，再把 log/ 拷回来")
        return

    print(f"[traction] {root}  共 {len(files)} 份 500 Hz 记录\n")
    for path in files:
        try:
            data, idx = load(path)
        except (ValueError, FileNotFoundError) as exc:
            print("=" * 72)
            print(f"{path.name}   ✗ 跳过")
            print(f"  {exc}\n")
            continue
        duration = len(data) / rma_spec.JOINT_HZ
        standing = any(k in path.name for k in args.standing)
        print("=" * 72)
        print(f"{path.name}   {len(data):,} 行 = {duration:.1f} s"
              f"{'   [按静止段处理]' if standing else ''}")
        print("=" * 72)
        for block in (report_timing(data, idx), report_accel(data, idx),
                      report_quaternion(data, idx),
                      report_torque(data, idx, standing),
                      report_body_force(data, idx)):
            for line in block:
                print("  " + line)
            print()

        if standing:
            cols = [f"{p}_{n}" for p in ("q", "dq") for n in
                    [f"{leg}_{jt}" for leg in ("fl", "fr", "hl", "hr")
                     for jt in ("roll", "pitch", "knee")]]
            have = [c for c in cols if c in idx]
            if have:
                sigma = second_diff_sigma(data[:, [idx[c] for c in have]])
                half = len(have) // 2
                print(f"  噪声本底   关节角 σ ≈ {np.median(sigma[:half]):.5f} rad   "
                      f"关节速度 σ ≈ {np.median(sigma[half:]):.4f} rad/s")
                print("             （仿真采集的观测噪声按这个量级注入，"
                      "不要用 IsaacLab 四足的通用值）\n")


if __name__ == "__main__":
    main()
