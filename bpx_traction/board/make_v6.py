#!/usr/bin/env python3
"""从板子的 v4 控制脚本生成 v6（牵引数据采集版）。

    python bpx_traction/board/make_v6.py 原始的_bpx_dual_pad.py 输出_bpx_dual_pad.py

**为什么用补丁脚本而不是手改一份**：锚点找不到就直接报错退出，不会悄悄生成一个改错的
文件。板子上那份 v4 和本地存档可能有出入（版本、本地改动），手改很容易对着错的行改。

## 改了什么（三处，全是加法）

1. `read_high_snapshot()` 多读两个量：
   `getJointTorqueHighRate()`（12 维关节力矩）和 `getImuAccHighRate()`（3 维加速度）。
   这 15 维是力估计器 61 维输入里**唯一在真机上没验证过**的部分 —— 现有的 500 Hz
   记录器只记 `tff`（前馈指令，策略模式下恒为 0），历史记录里一帧真实力矩都没有。

2. `REC_JOINT_COLS` 加 16 列：12 力矩 + 3 加速度 + 1 个 `snap_us`。

3. `Recorder.joint()` 把这些写进去。

`snap_us` 是**这次快照读取花了多少微秒**。500 Hz 的预算是 2000 µs，加两个 getter 之后
还剩多少余量，这个数直接回答。把 `RECORD_EXTRA_SENSORS` 翻成 False 再跑一段，
两边 `snap_us` 一减就是那两个 getter 的净开销。

## 不改什么

**控制逻辑一行不动。** `build_obs_frame` / `run_policy` / `make_policy_session` /
`POLICY_TERM_SLICES` / PD / 门 2 全部原样。v6 跑起来的行为和 v4 完全一样，
只是多记了 16 列数据。

所以 v6 **不是**牵引功能版，是**数据采集版**。牵引层上板是后面的事，要等估计器在真机
数据上重新验证过、门 3 做好。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 锚点 → 替换。每一条都必须在源文件里**恰好出现一次**，否则退出。
# --------------------------------------------------------------------------
PATCHES: list[tuple[str, str, str]] = [

    ("在 REC_JOINT_COLS 末尾加 16 列",
     """                  + ['cmd_vx', 'cmd_vy', 'cmd_wz'] + ['ptarget_' + n for n in _RJ])""",
     """                  + ['cmd_vx', 'cmd_vy', 'cmd_wz'] + ['ptarget_' + n for n in _RJ]
                  # --- v6 牵引数据采集：力估计器 61 维输入里真机没验过的那 15 维 ---
                  + ['tau_' + n for n in _RJ] + ['acc_x', 'acc_y', 'acc_z'] + ['snap_us'])
RECORD_EXTRA_SENSORS = True   # 翻 False 跑一段，和 True 那段的 snap_us 相减 = 两个新 getter 的净开销"""),

    ("snapshot 多读力矩和加速度，并计时",
     """                    seq0 = int(client.getJointStateSeqHighRate())
                    q = vec(client.getJointPositionHighRate(), 12, 'joint_position_high_rate')
                    dq = vec(client.getJointVelocityHighRate(), 12, 'joint_velocity_high_rate')
                    omega = vec(client.getImuOmegaHighRate(), 3, 'imu_omega_high_rate')
                    quat = vec(client.getImuQuatHighRate(), 4, 'imu_quat_high_rate')
                    robot_ts = client.getJointStateTimestampHighRate()
                    seq1 = int(client.getJointStateSeqHighRate())""",
     """                    seq0 = int(client.getJointStateSeqHighRate())
                    _snap_t0 = mono()
                    q = vec(client.getJointPositionHighRate(), 12, 'joint_position_high_rate')
                    dq = vec(client.getJointVelocityHighRate(), 12, 'joint_velocity_high_rate')
                    omega = vec(client.getImuOmegaHighRate(), 3, 'imu_omega_high_rate')
                    quat = vec(client.getImuQuatHighRate(), 4, 'imu_quat_high_rate')
                    # --- v6 新增：力估计器要的两组量 ---
                    if RECORD_EXTRA_SENSORS:
                        tau = vec(client.getJointTorqueHighRate(), 12, 'joint_torque_high_rate')
                        acc = vec(client.getImuAccHighRate(), 3, 'imu_acc_high_rate')
                    else:
                        tau, acc = np.full(12, np.nan), np.full(3, np.nan)
                    robot_ts = client.getJointStateTimestampHighRate()
                    seq1 = int(client.getJointStateSeqHighRate())
                    _snap_us = (mono() - _snap_t0) * 1e6"""),

    ("snapshot 字典带上新字段",
     """                snap = {
                    'seq': seq1, 'q': q, 'dq': dq, 'omega': omega, 'quat': quat,
                    'robot_ts': robot_ts, 'arrival': now,
                }""",
     """                snap = {
                    'seq': seq1, 'q': q, 'dq': dq, 'omega': omega, 'quat': quat,
                    'robot_ts': robot_ts, 'arrival': now,
                    'tau': tau, 'acc': acc, 'snap_us': _snap_us,   # v6
                }"""),

    ("记录器把新字段写进 500 Hz 那一行",
     """                snap['q'], snap['dq'], snap['quat'], snap['omega'],
                command[0], command[1], command[2], command[3], command[4], self.cmd, self.target]))""",
     """                snap['q'], snap['dq'], snap['quat'], snap['omega'],
                command[0], command[1], command[2], command[3], command[4], self.cmd, self.target,
                # --- v6 ---
                snap.get('tau', np.full(12, np.nan)), snap.get('acc', np.full(3, np.nan)),
                [snap.get('snap_us', float('nan'))]]))"""),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="从 v4 生成 v6（牵引数据采集版）")
    ap.add_argument("source", type=Path, help="板子上那份 bpx_dual_pad.py")
    ap.add_argument("output", type=Path, help="输出路径")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出")
    args = ap.parse_args()

    if not args.source.is_file():
        print(f"找不到源文件：{args.source}", file=sys.stderr)
        return 1
    if args.output.exists() and not args.force:
        print(f"输出已存在（要覆盖加 --force）：{args.output}", file=sys.stderr)
        return 1

    text = args.source.read_text()
    source_sha = hashlib.sha256(text.encode()).hexdigest()

    for name, anchor, replacement in PATCHES:
        count = text.count(anchor)
        if count != 1:
            print(f"✗ 「{name}」的锚点在源文件里出现了 {count} 次，期望 1 次。", file=sys.stderr)
            print("  板子上那份和这个补丁对不上 —— 不要手改，先把两边的版本核对清楚。",
                  file=sys.stderr)
            return 2
        text = text.replace(anchor, replacement)
        print(f"✓ {name}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)

    print()
    print(f"源   {args.source}")
    print(f"     sha256 {source_sha[:16]}…")
    print(f"输出 {args.output}")
    print(f"     sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}…")
    print()
    print("控制逻辑一行没动 —— build_obs_frame / run_policy / make_policy_session /")
    print("POLICY_TERM_SLICES / PD / 门2 全部原样。v6 的行为和 v4 完全一样，只多记 16 列。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
