"""从 IsaacLab 状态拼 61 维估计器帧，以及 15 维标签。

## 为什么大部分量不从 Imu 传感器取

`imu_acc_b` 必须用 Imu 传感器 —— 它给的是**比力**（含重力），和真 IMU 一致，
而 `ArticulationData` 只有运动学加速度。

但角速度、投影重力、四元数**不**走 Imu 传感器，走 `ArticulationData`，因为
`bpx_rma` 的 45 维策略帧用的就是那几个（`mdp.base_ang_vel` / `mdp.projected_gravity`）。
估计器和策略在板子上读的是同一个 IMU，所以仿真里也该是同一个来源 —— 两边逐位相同，
而不是"两个几乎一样的量"。:func:`assert_matches_policy_frame` 把这件事变成断言。

`last_action` 同理直接读 `env.action_manager.action`，和 `mdp.last_action` 同一个张量。

## ⚠️ 已知的 sim2real 差距：IMU 加速度的冲击尖峰

实测采到的 `imu_acc_b`：均值 `[-0.5, 0.4, 10.3]`（站立的 9.81 ✓），但 z 轴 p99 到
65 m/s²、峰值 171 m/s²（17 g）。那是触地冲击在**有限差分**加速度上的尖峰。

真 IMU 不会这样 —— 它有模拟抗混叠滤波和有限带宽，冲击会被抹平得多。所以估计器有可能
学会依赖一类真机上不存在的特征。

这一条**只能用真机的加速度记录来判**，而板子现在的 500 Hz 记录器不记 acc
（只记 `tff`）。等那批数据到手，对比两边的 acc 谱，必要时在仿真侧加一个匹配真机带宽的
低通。在那之前这是个挂账的风险，不是已解决的问题。

## 关节顺序

IsaacLab 内部的关节顺序是交错的，和 SDK 顺序完全不同。所有按关节的量都用
`preserve_order=True` 解析出来的索引重排到 `spec.JOINT_NAMES`。这一条错了，
12 个关节会静默接错，训练损失照样好看，上真机才发现。
"""

from __future__ import annotations

import torch

from .. import spec


class FrameAssembler:
    """把一个 `ManagerBasedRLEnv` 的当前状态拼成 61 维帧 + 15 维标签。"""

    def __init__(self, env, imu_name: str = "imu", robot_name: str = "robot"):
        self.env = env
        self.robot = env.scene[robot_name]
        self.device = self.robot.device

        # 关节：SDK 顺序，preserve_order 保证逐位对齐 spec.JOINT_NAMES
        joint_ids, joint_names = self.robot.find_joints(spec.JOINT_NAMES, preserve_order=True)
        if list(joint_names) != list(spec.JOINT_NAMES):
            raise RuntimeError(
                f"关节顺序解析错了：\n  拿到 {list(joint_names)}\n  期望 {list(spec.JOINT_NAMES)}"
            )
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.default_joint_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone()

        # IMU：只用它的 lin_acc_b（比力，含重力）
        self.imu = env.scene.sensors.get(imu_name) if hasattr(env.scene, "sensors") else None
        if self.imu is None:
            try:
                self.imu = env.scene[imu_name]
            except KeyError as exc:
                raise RuntimeError(
                    f"场景里没有名为 {imu_name!r} 的 Imu 传感器 —— 估计器输入的前 3 维是"
                    f"含重力的 IMU 加速度，拿不到就没法采。用 bpx_traction.env.traction_cfg "
                    f"里的配置，或自己加一个 ImuCfg(gravity_bias=(0,0,9.81))。"
                ) from exc

        # 足底接触：标签用
        foot_ids, _ = env.scene["contact_forces"].find_bodies(
            spec.rma_spec.FOOT_BODIES, preserve_order=True
        )
        self.foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=self.device)
        self.contact_sensor = env.scene["contact_forces"]

        #: 整机重量，足底负载按它归一。
        self.weight_n = float(self.robot.data.default_mass.sum(dim=1)[0]) * 9.81

    # ----------------------------------------------------------------------
    def frame(self) -> torch.Tensor:
        """当前帧 `(N, 61)`，分段顺序由 `spec.TERMS` 定义。"""
        data = self.robot.data
        parts = [
            self.imu.data.lin_acc_b,                                   # 3  含重力的比力
            data.root_ang_vel_b,                                       # 3  和策略帧同源
            data.projected_gravity_b,                                  # 3  同上
            data.root_quat_w,                                          # 4  wxyz
            data.joint_pos[:, self.joint_ids] - self.default_joint_pos,  # 12
            data.joint_vel[:, self.joint_ids],                         # 12
            data.applied_torque[:, self.joint_ids],                    # 12
            self.env.action_manager.action,                            # 12 和 mdp.last_action 同源
        ]
        out = torch.cat(parts, dim=1)
        if out.shape[1] != spec.FRAME_SIZE:
            raise RuntimeError(
                f"拼出来 {out.shape[1]} 维，契约是 {spec.FRAME_SIZE} 维。"
                f"各段宽度：{[p.shape[1] for p in parts]}"
            )
        return out

    def labels(self, force_b: torch.Tensor) -> torch.Tensor:
        """15 维标签 `(N, 15)`。`force_b` 由 `TetherApplier.apply_world()` 返回。

        **只有 force 这一组有消费端**，其余四组是辅助任务 —— 它们是共享主干的正则项，
        删掉会改变主干学到的表示（参考实现给 height 的损失权重高达 8.0）。
        """
        data = self.robot.data
        contact_force = self.contact_sensor.data.net_forces_w[:, self.foot_ids]   # (N, 4, 3)
        magnitude = torch.linalg.norm(contact_force, dim=-1)                      # (N, 4)

        # 足底负载按整机重量归一，并**夹到头的量程上界**。
        # 实测 p50=0.18、p99=0.76、p99.9=2.08，只有 0.13% 的样本超过 2.0 —— 那是触地
        # 冲击的瞬时尖峰。不夹的话标签会落在 sigmoid×2.0 够不到的地方，留下一截消不掉的
        # 损失；把头放宽到 4.0 又会把 p50=0.18 这些典型值挤进 sigmoid 的饱和低区，
        # 反而丢分辨率。这是个辅助头（只用来正则主干），夹掉尖峰不损失任何有用信号。
        load_scale = spec.OUTPUT_ACTIVATIONS["foot_load"][1]
        load = torch.clamp(magnitude / self.weight_n, max=load_scale)

        parts = [
            data.root_lin_vel_b,                                   # 3  机身系线速度
            data.root_pos_w[:, 2:3],                               # 1  高度（平地 = 世界 z）
            (magnitude > 1.0).float(),                             # 4  触地（和 mdp.feet_contact 同阈值）
            load,                                                  # 4  足底负载，按体重归一
            force_b,                                               # 3  ★ 外力，机身系
        ]
        out = torch.cat(parts, dim=1)
        if out.shape[1] != spec.OUTPUT_SIZE:
            raise RuntimeError(
                f"标签拼出来 {out.shape[1]} 维，契约是 {spec.OUTPUT_SIZE} 维。"
                f"各段宽度：{[p.shape[1] for p in parts]}"
            )
        return out

    # ----------------------------------------------------------------------
    def assert_matches_policy_frame(self, policy_obs: torch.Tensor, atol: float = 0.0) -> None:
        """估计器帧和策略帧的**公共项必须逐位相同**。

        两者共享 5 组量：关节位置、关节速度、角速度、投影重力、上一动作。它们在板子上
        来自同一次传感器读取，仿真里也必须同源。这条断言能当场抓住关节顺序接错、
        默认位姿减错、`last_action` 取了不同时刻这类静默 bug —— 那些错误不会让训练崩，
        只会让估计器学到一个偏的映射，上真机才暴露。

        `policy_obs` 给 `obs["policy"]` 的当前帧 `(N, 45或47)`。
        """
        frame = self.frame()
        pairs = [
            ("joint_pos", "joint_pos_rel"),
            ("joint_vel", "joint_vel"),
            ("base_ang_vel", "imu_gyro_b"),
            ("projected_gravity", "projected_gravity_b"),
            ("last_action", "last_action"),
        ]
        policy_slices = dict(
            zip([name for name, _ in spec.rma_spec.OBS_TERMS], spec.rma_spec.obs_term_slices())
        )
        problems = []
        for policy_term, our_term in pairs:
            pb, pe = policy_slices[policy_term]
            ob, oe = spec.term_slice(our_term)
            lhs, rhs = policy_obs[:, pb:pe], frame[:, ob:oe]
            if atol > 0:
                ok = torch.allclose(lhs, rhs, atol=atol)
            else:
                ok = torch.equal(lhs, rhs)
            if not ok:
                diff = (lhs - rhs).abs().max().item()
                problems.append(
                    f"  {policy_term}[{pb}:{pe}] vs {our_term}[{ob}:{oe}]  最大差 {diff:.3e}"
                )
        if problems:
            raise AssertionError(
                "估计器帧和策略帧的公共项对不上（多半是关节顺序或默认位姿）：\n"
                + "\n".join(problems)
            )

    def describe(self) -> str:
        return (
            f"61 维帧：IMU 加速度取自传感器（含重力），角速度/投影重力/四元数/上一动作"
            f"与策略帧同源\n"
            f"关节顺序 {spec.JOINT_NAMES[0]} … {spec.JOINT_NAMES[-1]}（SDK，preserve_order）\n"
            f"足底负载按整机重量 {self.weight_n:.1f} N 归一"
        )
