"""BPX-Pro 平地速度跟踪环境（RMA 师生蒸馏用）。

三个观测组，对应部署契约和蒸馏的三路输入：

====================  ====  ==========================================================
组                     维度   内容
====================  ====  ==========================================================
``policy``             45    板子上 `build_obs_frame()` 的那一帧，顺序严格按
                             `POLICY_TERM_SLICES`
``history``          1125    同样 6 项，每项 25 帧历史、term-major 展平；等于板子上
                             `flatten_term_major_history()` 的输出
``privileged``         ~62   辨识出来的那些真值（armature / 粘滞 / 摩擦 / 零偏 /
                             延迟 / 质量 / 膝负载）+ 机身线速度 + 触地
====================  ====  ==========================================================

student 吃 ``policy + history``（1170），teacher 吃 ``policy + privileged``。
teacher 的权重来自先跑的 PPO（PPO 的 actor 观测集必须和 teacher 一模一样）。

**为什么默认不加观测噪声**：板子上 history[-1] 就是同一帧 `policy`，两者逐位相等。
IsaacLab 里 ``policy`` 和 ``history`` 是两个独立的组，各自算噪声的话这个恒等关系
就断了。关掉噪声时两组都是同一状态的确定性函数，自动逐位相等。要加噪声必须先做
一个"每步只采样一次、两组共享"的缓存，不要只把 `enable_corruption` 打开。
"""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp, spec
from .params import get_params
from .robot_cfg import make_robot_cfg

# 所有按关节的项都要用 SDK / 策略顺序，不能让 IsaacLab 按它自己的关节顺序来
ROBOT_SDK_JOINTS = SceneEntityCfg("robot", joint_names=spec.JOINT_NAMES, preserve_order=True)


##
# 场景
##
@configclass
class BpxProSceneCfg(InteractiveSceneCfg):
    """平地 + BPX-Pro。板子上跑的是 flat 策略，没有高度图，所以这里也不放 height scanner。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot = make_robot_cfg()

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


##
# MDP
##
@configclass
class CommandsCfg:
    """速度指令。上限参考板子上的 `MOTION_CAPS`（vx 2.0 / vy 0.6 / yaw 1.5）和真机走路记录
    里实际出现过的 vx ≈ 1.43 m/s，留了余量但不超过手柄能发的范围。

    `heading_command=False`：板子直接发 wz，不是发朝向。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class ActionsCfg:
    """关节位置：target = q0 + 0.1875 · raw，和板子上 `run_policy()` 一致。

    raw 的 ±4 裁剪由 rsl_rl 的 wrapper 做（agent cfg 里 `clip_actions=4.0`），
    和板子上的 `np.clip(network_action, ±ACTION_CLIP)` 对应。
    板子还会把 target 裁到 `MODEL_LIMITS`，但 q0 ± 0.1875·4 = ±0.75 落在限位内侧，
    那个裁剪永远不会触发，所以仿真里不用再加。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=spec.JOINT_NAMES,
        preserve_order=True,
        scale=spec.ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class PolicyFrameCfg(ObsGroup):
    """板子上 `build_obs_frame()` 的 45 维，顺序不能动。"""

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ROBOT_SDK_JOINTS})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ROBOT_SDK_JOINTS})
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    last_action = ObsTerm(func=mdp.last_action)
    velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    def __post_init__(self):
        # 见模块 docstring：加噪声会破坏 policy 和 history 的逐位相等关系
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class HistoryFrameCfg(PolicyFrameCfg):
    """同样 6 项、每项 25 帧，term-major 展平 = 1125 维。

    `CircularBuffer.buffer` 是"最旧在前、最新在后"，`flatten_history_dim` 按时间做外层展开，
    每一项展成 (25 × dim)，再按项拼接 —— 和板子上
    ``concat([history[:, b:e].reshape(-1) for b, e in POLICY_TERM_SLICES])`` 逐位相同。
    """

    def __post_init__(self):
        super().__post_init__()
        self.history_length = spec.HISTORY_LENGTH
        self.flatten_history_dim = True


# --- 相位进观测（Sprint6 用）：在 45 维末尾追加 [sinφ, cosφ]，前 45 维一字不动 ---
# 板子将来复现时只需在帧末尾多算这 2 维时钟，前 45 维逻辑完全不变。周期按我们
# 自己的步频 2.2Hz 定（GAIT_PERIOD=0.45），和 feet_gait 奖励同一个时钟。
_PHASE_TERM = dict(func=mdp.gait_phase_obs, params={"period": 0.45, "command_name": "base_velocity"})


@configclass
class PhasePolicyFrameCfg(PolicyFrameCfg):
    """45 维 + 相位 2 维 = 47 维。phase 作为第 7 项追加在末尾。"""

    phase = ObsTerm(**_PHASE_TERM)


@configclass
class PhaseHistoryFrameCfg(HistoryFrameCfg):
    """47 维 × 25 帧 = 1175 维。"""

    phase = ObsTerm(**_PHASE_TERM)


@configclass
class PrivilegedCfg(ObsGroup):
    """只给 teacher。这些就是辨识产出的量——student 得从历史里把它们猜出来。"""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    armature_ratio = ObsTerm(func=mdp.actuator_armature_ratio)
    viscous_ratio = ObsTerm(func=mdp.actuator_viscous_ratio)
    friction_ratio = ObsTerm(func=mdp.actuator_friction_ratio)
    knee_friction = ObsTerm(func=mdp.knee_friction_abs)
    knee_load = ObsTerm(func=mdp.knee_load)
    encoder_bias = ObsTerm(func=mdp.encoder_bias)
    delay_steps = ObsTerm(func=mdp.torque_delay_steps)
    mass_offset = ObsTerm(func=mdp.mass_offset)
    feet_contact = ObsTerm(
        func=mdp.feet_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.FOOT_BODIES)},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ObservationsCfg:
    policy: PolicyFrameCfg = PolicyFrameCfg()
    history: HistoryFrameCfg = HistoryFrameCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """域随机化。关节参数和质量的区间都来自 `data/bpx_pro_training_params.yaml`。"""

    # --- startup ---
    ground_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[spec.TORSO_BODY]),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # --- reset ---
    #: armature / 粘滞 / 库伦摩擦 / 编码器零偏 / 力矩延迟，全部按辨识区间抽
    actuator_params = EventTerm(func=mdp.randomize_actuator_params, mode="reset")
    #: 整机质量 15.1 ± 1 kg（USD 是 17.55）
    #: asset_cfg 必须写在 params 里 —— EventManager 只解析 params 里的 SceneEntityCfg，
    #: 函数签名上的默认值不会被解析出 body_ids
    total_mass = EventTerm(
        func=mdp.randomize_total_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[spec.TORSO_BODY])},
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )

    # --- interval ---
    #: 方案 B：每个控制步按负载更新膝摩擦 F = f0 + mu·L。
    #: `interval_range_s` 在 `__post_init__` 里设成 step_dt，保证每步都触发。
    knee_friction = EventTerm(
        func=mdp.knee_load_friction, mode="interval", interval_range_s=(0.02, 0.02), is_global_time=False
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    # -- 任务
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- 惩罚
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(
        func=mdp.base_height_l2, weight=-1.0, params={"target_height": 0.40}
    )
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.FOOT_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=spec.THIGH_BODIES + spec.CALF_BODIES),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[spec.TORSO_BODY]), "threshold": 1.0},
    )


##
# 环境
##
@configclass
class BpxProFlatEnvCfg(ManagerBasedRLEnvCfg):
    """50 Hz 控制 / 500 Hz 仿真，和真机的策略频率、PD 内环频率一致。"""

    scene: BpxProSceneCfg = BpxProSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = spec.DECIMATION          # 10
        self.episode_length_s = 20.0
        self.sim.dt = spec.SIM_DT                  # 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # 膝摩擦负载模型必须每个控制步都跑一次
        # （cfg 上没有 step_dt，那是 env 的属性，这里自己算）
        step_dt = self.decimation * self.sim.dt
        self.events.knee_friction.interval_range_s = (step_dt, step_dt)

        # yaml 里关掉负载模型就退化成方案 A（常数摩擦，每回合抽一次），
        # 这时候不需要每步事件，直接去掉省开销
        if not get_params().knee_load_friction.enabled:
            self.events.knee_friction = None

        self.viewer.eye = (2.0, 2.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)


@configclass
class BpxProFlatFastEnvCfg(BpxProFlatEnvCfg):
    """高速版：vx 上限 3.0 m/s。

    **除了指令范围，和 `BpxProFlatEnvCfg` 完全一样** —— 奖励权重、随机化、网络、频率都不动。
    改奖励会让这一组和已有的 1.5 m/s 两组不可比；已知的瑕疵（`feet_air_time` 的
    threshold 0.4 s 对这个机器人偏高，该项一直在扣分）在所有组里同样存在，不构成偏差。

    **上限怎么定的**：不是手柄（板子的 `MOTION_CAPS` 可以改），是电机。
    1.5 m/s 实测关节峰值速度膝 5.7 / 髋 9.6 rad/s（限位 15 / 22），线性外推到 3.0 是
    膝 11.4（76%）/ 髋 19.2（87%）；此时 DCMotor 的力矩-速度曲线把膝可用力矩压到
    约 24 N·m，已低于 30 N·m 的部署限幅 —— 也就是 3.0 m/s 附近电机本身成为瓶颈。
    参照同量级的 Unitree A1（3.3 m/s）/ Go1（3.7 m/s），3.0 是够得着的目标。

    **动机**：直线 1.5 m/s 下辨识组和厂家组跟踪质量打平（adaptation module 把参数误差
    在线吸收掉了）。膝惯量占关节力矩的比例随速度急剧上升（0.5 m/s 时 0.9%，1.5 m/s 时
    9.4%），高速下还会更高 —— 这一档有可能把分辨力重新打开。

    注意：上真机跑这个速度需要同时放开板子 `bpx_dual_pad.py` 里的
    ``MOTION_CAPS = {'vx': 2.0, 'vy': 0.6, 'yaw': 1.5}``，否则手柄发不出超过 2.0 的指令。
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)


@configclass
class BpxProFlatSprintEnvCfg(BpxProFlatFastEnvCfg):
    """冲刺版：指令上限 4.5 m/s，并松开三个压制奔跑步态的奖励项。

    **这一组和辨识对比的四组不可比**（奖励不同），它是纯粹的速度探索，不进对照实验。

    为什么不是单纯把指令调高：当前小跑步态的"膝速度 / 体速"比值在恶化
    （3.93 rad/s 每 m/s @3.0 → 4.17 @3.5），说明这个步态本身不适合更高速。
    官方手册 7.2 列的步态里有 `setRunning()` / `setBound()` / `setPronk()`，
    4 m/s 的标称多半靠奔跑或跳跃步态达成 —— 那类步态每米所需的膝摆动次数更少。
    下面三项正好是在阻止策略往那个方向走：

    * ``flat_orientation_l2`` −1.0 → −0.2：奔跑必然有俯仰起伏，原权重在罚它
    * ``base_height`` 目标 0.40 → 0.32：高速下策略自己就蹲到 0.30，强拉回 0.40 白耗奖励
    * ``feet_air_time`` threshold 0.4 → 0.25 s：这个机器人达不到 0.4 s 腾空，
      该项一直在扣分（早在 1.5 m/s 组就发现了），高速下步频更快、更严重

    电机包络按官方手册：最大关节扭矩 33 N·m（见 `spec.EFFORT_LIMIT`）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 4.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.8, 0.8)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)

        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height.params["target_height"] = 0.32
        self.rewards.feet_air_time.params["threshold"] = 0.25


@configclass
class BpxProFlatSprint2EnvCfg(BpxProFlatSprintEnvCfg):
    """Sprint 第二版：vx 上限 4.0，修站立抖动 + 建模 dq 滤波滞后。

    针对 2026-04-18（板子时钟）真机实测的两种抖动：

    * **2 Hz 整机摆**（辨识组，仿真也有 = 训练出来的极限环）：
      零指令环境 5% → 20%，新增 stand_still 惩罚，动作变化率 −0.01 → −0.05
    * **9.3 Hz 网络输出振荡**（baseline 组真机独有，仿真没有 = 未建模的板载
      dq 滤波滞后激发）：PaceDCMotor 内加一阶低通（500 Hz 更新，PD 阻尼和
      策略观测用同一个滤波值，位置与真机一致），时间常数每回合随机 2–10 ms

    其余沿用 sprint 的奖励放松（姿态 −0.2 / 高度 0.32 / 腾空 0.25——sprint 的
    仿真站立反而是三个高速策略里最稳的，这组放松无罪）。
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 4.0)   # 用户要求上限 4.0
        self.commands.base_velocity.rel_standing_envs = 0.20

        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.5,
            params={"command_name": "base_velocity", "command_threshold": 0.06},
        )

        # 观测里的关节速度换成执行器滤波值（不开滤波时逐位等于原值）
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_filtered)
        self.observations.history.joint_vel = ObsTerm(func=mdp.joint_vel_filtered)
        # 滤波时间常数随机化
        self.events.actuator_params.params = {"dq_filter_tau_range": (0.002, 0.010)}


@configclass
class BpxProFlatSprint3EnvCfg(BpxProFlatSprint2EnvCfg):
    """Sprint 第三版：在 sprint2 基础上修"蜘蛛腿"（髋外展宽站位）。

    定量依据（MuJoCo corrected，2026-10-14）：sprint2 行走时髋 roll 外展均值
    8.9° @1.5 m/s、21.3°（峰值 41°）@3.0，左右脚距 0.51→0.74 m（机械髋距 0.36）；
    而 flat / fast 同样的推搡随机化下站位正常 —— 元凶是 sprint2 自己的
    低高度目标(0.32) + 松姿态惩罚(−0.2)，蹲得低就撇得开。

    两处修正，其余（站立修复、dq 滞后建模、vx 4.0）全部保留：
      * 髋 roll 偏离 q0 的 L1 惩罚，权重 −0.5，**全程生效**（不只站立时）
      * 高度目标 0.32 → 0.37（真机站姿区间下沿）

    预期代价：宽站位的抗侧向干扰没了，3.0 附近的稳定性要重新验。
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.base_height.params["target_height"] = 0.37
        self.rewards.hip_roll_deviation = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
        )


@configclass
class BpxProFlatSprint4EnvCfg(BpxProFlatSprint3EnvCfg):
    """Sprint 第四版：在 sprint3 基础上修"抬腿太低" + 扩倒车范围。

    真机反馈（sprint3，2026-10）：平地贴地掠过，地面有纹理就蹭脚；倒车只训到
    −1.0 m/s，手柄给大了就 OOD。抬腿低不是 bug —— 仿真地面绝对平整时贴地
    是能耗/action-rate 最优解，feet_air_time 只管摆动时长不管高度。

    两处修正，其余（站立修复、dq 滞后、髋 roll、高度 0.37、vx 4.0）全部保留：
      * foot_clearance 惩罚：摆动相 toe_link 世界 z 偏离 0.10 扣分
        （≈ 离地 7 cm，FOOT_RADIUS 0.0285），足端水平速度 tanh 门控不扰支撑脚
      * 倒车 −1.0 → −2.0 m/s（手柄上限 −3，但高速倒车没有需求，先给出余量）

    **从零训练三连败的教训（都删库了，这里留档）**：
      v1 Spot exp 正奖励 → 脚不动门控 0、exp(0) 满分，集体站桩；
      v2 恒定 −5 惩罚 → 压死早期迈步探索，慢腾挪（1.5 m/s 误差 0.53）；
      v3 课程 −0.5→−3.0 → 还是慢腾挪（误差 1.07）。根因：速度门控有第二条
      逃逸路径——"把脚挥慢"也能消门控，从零训练总是先找到这条捷径，
      起飞期（本配置系在 1100+ 迭代才起飞）被它锁死。
    所以 sprint4 **不从零训**：种入 sprint3 已收敛 teacher（跟踪 1.25）续训
    800 迭代。已有步态下挥慢脚会立刻损失大额跟踪奖励，逃逸路径被既得利益
    堵死，惩罚只能走"抬高脚"这条正路。

    预期代价：抬腿高了腾空动量变大，能耗和姿态平稳度略差；4.0 冲刺步态要重验。
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 4.0)
        # 二次型 foot_clearance_penalty 在 -2 和 -6 下真峰值都卡在 2.5~3 cm（平台期），
        # 换单边线性亏空（见 rewards.foot_clearance_deficit 注释）。-1.5 × 500 迭代：
        # 真峰值 3.0→3.4 cm 且跟踪无损；-3.0 × 600：6.2 cm 达标 —— 这是最终值。
        #
        # BPX_S4_CLEARANCE 环境变量（如 "quad:-2.0" / "lin:-1.5"）只给对照组复刻
        # 辨识组的四轮剂量路径用（quad-2 → quad-6 → lin-1.5 → lin-3），平时不设 =
        # 最终配置，行为不变。
        _clear = os.environ.get("BPX_S4_CLEARANCE", "lin:-3.0")
        _form, _w = _clear.split(":")
        self.rewards.foot_clearance = RewTerm(
            func=mdp.foot_clearance_deficit if _form == "lin" else mdp.foot_clearance_penalty,
            weight=float(_w),
            params={
                "tanh_mult": 2.0,
                "target_height": 0.10,
                "asset_cfg": SceneEntityCfg("robot", body_names=spec.FOOT_BODIES),
            },
        )

        # 横滚专项惩罚：sprint 把 flat_orientation 放松到 -0.2 做高速俯仰，连带松了
        # 横滚，随机对称破缺长到 ~2°（真机 roll 偏置 -1.44°，flat/fast 只 <1°）。
        # 单独重罚横滚(grav_y²)把它压回去，不动俯仰放松。见 rewards 注释。
        self.rewards.flat_roll = RewTerm(func=mdp.flat_orientation_roll_l2, weight=-2.0)


@configclass
class BpxProFlatSprint4EnvCfg_PLAY(BpxProFlatSprint4EnvCfg):
    """Sprint4 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint5EnvCfg(BpxProFlatSprint4EnvCfg):
    """Sprint 第五版：在 sprint4(rollfix) 上**只加一个最小相位奖励**，其它一概不动。

    sprint4(rollfix) 的全部保留——横滚修复 flat_roll、抬脚 deficit(7cm)、站立修复、
    dq 滞后、髋 roll、高度 0.37、vx 4.0、倒车 -2.0，**一个都不换**。

    唯一新增：`feet_gait`——支撑相该触地、摆动相该离地的接触-相位匹配比例(正奖励)，
    让对角 trot 按一个节拍更规律地交替。相位是环境里的纯时间时钟，**只驱动这一个
    奖励、不进 45 维观测**，板子契约零改动。参数按我们自己的步态定：

      * 周期 T=0.45s ≈ 2.2Hz（我们实测步频 1.5→2.07 / 3.0→2.5 的中值，不是 Go2 的值）
      * duty=0.65（我们实测占空比 ~0.65，不是 Go2 文档的 0.52/0.55）
      * 权重 +0.5（温和引导，不强行改步态结构）

    rollfix 的 trot 本来就 0.80/0.90 很规律，这一项是锦上添花式的节拍规整，不期望
    大改，也不碰抬脚/横滚/速度。
    """

    GAIT_PERIOD = 0.45
    GAIT_DUTY = 0.65

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_gait = RewTerm(
            func=mdp.feet_gait,
            weight=0.5,
            params={"period": self.GAIT_PERIOD, "duty": self.GAIT_DUTY},
        )


@configclass
class BpxProFlatSprint5EnvCfg_PLAY(BpxProFlatSprint5EnvCfg):
    """Sprint5 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint6EnvCfg(BpxProFlatSprint5EnvCfg):
    """Sprint 第六版：**相位进观测**（45→47）+ 极速拉到 5.0 + 横移 ±2.0。

    在 sprint5(rollfix + feet_gait) 上加三件事，验证过的东西全保留、板子不碰：

      * **相位进观测**：45 维末尾追加 [sinφ, cosφ] 两维（见 PhasePolicyFrameCfg）。
        策略每步显式看到节拍——低速动作小、本体感觉弱时，这个显式节拍是走稳转稳
        的关键（光靠 sprint5 的 feet_gait 奖励低速没改善，就是缺这个显式输入）。
        前 45 维一字不动，板子将来只需末尾多算 2 维时钟。
      * **极速 4.0 → 5.0**：lin_vel_x 上限拉到 5.0，让策略见过更高速。
      * **横移 ±0.8 → ±2.0**：lin_vel_y 大幅加大。

    注意这版**先不加能量门控**——先单独验证"相位进观测对低速的作用"和"拉到 5.0
    自然能跑多快"。极速若上不去，下一版再加高速放松力矩/加速度惩罚的门控。

    观测 47 维 ⇒ 导出 47 维 ONNX，**和现有 45 维板子契约不兼容**（check_contract
    的 45 维断言对这版会失败，是预期的）；要上板得先把板子改成 47 维。
    """

    def __post_init__(self):
        super().__post_init__()
        # 相位进观测：换成 47 维的观测组（前 45 维和 sprint5 完全一致，末尾加相位）
        self.observations.policy = PhasePolicyFrameCfg()
        self.observations.history = PhaseHistoryFrameCfg()
        # 极速冲 5.0、横移 ±2.0；倒车保持 -2.0，转向保持 ±2.0
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 5.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)


@configclass
class BpxProFlatSprint6EnvCfg_PLAY(BpxProFlatSprint6EnvCfg):
    """Sprint6 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint7EnvCfg(BpxProFlatSprint4EnvCfg):
    """Sprint 第七版：**能量门控冲极速**。在 rollfix(sprint4) 上,45 维不变、板子不碰。

    sprint6 验证了:相位进观测没带来低速改善、极速反而略慢。所以这版回到 rollfix
    这个已验证的最好基底(45 维,抬脚/横滚/辨识全保留),只做一件事——**破极速的
    "站桩天花板"**。

    rollfix 极速卡在 ~3.8:指令超过能力时策略消极站桩,因为"拼命冲拿不到跟踪分、还
    挨一堆力矩罚"不如"站着少挨罚"。根因是力矩/加速度/动作率惩罚**恒定**,高速时比
    跟踪奖励还重,把"发力"算成了亏本。

    **能量门控**:这三个惩罚的权重随指令速度放松(见 rewards._speed_gate)——
      * |cmd_vx| ≤ 2.0:全额惩罚(低速省能平稳,不动 rollfix 的好处);
      * |cmd_vx| ≥ 4.5:降到 10-20%(高速几乎不罚,让策略敢用大力矩、快动作去冲);
      * 之间线性。
    外加极速 4.0 → 5.0,让策略见过并去够更高速。

    预期:低速和 rollfix 一样(门控低速=全额,等价),高速突破 3.8。代价是高速能耗、
    姿态平稳度下降(放开了嘛),但那正是冲速度该付的。
    """

    GATE = {"v_lo": 2.0, "v_hi": 4.5}

    def __post_init__(self):
        super().__post_init__()
        # 三个能量惩罚换成速度门控版（同权重，高速放松）
        self.rewards.dof_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2_gated, weight=-2.0e-5,
            params={**self.GATE, "gate_min": 0.1},
        )
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2_gated, weight=-2.5e-7,
            params={**self.GATE, "gate_min": 0.1},
        )
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2_gated, weight=-0.05,
            params={**self.GATE, "gate_min": 0.2},
        )
        # 极速拉到 5.0（横移保持 rollfix 的 ±0.8，不动，专注冲前向极速）
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 5.0)


@configclass
class BpxProFlatSprint8EnvCfg(BpxProFlatSprint7EnvCfg):
    """Sprint 第八版：**极速强化**。在 sprint7 能量门控基础上把旋钮全调开,冲 5.0-5.5。

    sprint7 的门控把极速从 3.8 推到 ~4.5。这版继续拧:高速残余惩罚更低、放松区
    延伸更高、指令上限拉到 6.0,让策略在高速段几乎零惩罚、敢彻底发力。

      * gate_min 0.1/0.2 → **0.03**(高速残余惩罚砍到 3%,几乎不罚)
      * v_hi 4.5 → **5.5**(放松区延伸到 5.5,让 4.5-5.5 段也放开)
      * 指令上限 5.0 → **6.0**(见过并去够更高速)

    低速段(≤v_lo=2.0)门控仍是全额,和 rollfix/sprint7 一样,不动低速的好处。
    物理天花板:电机 33 N·m 的功率上限,高速功率=力矩×转速,到某点电机饱和,
    门控放再开也没用——所以目标 5.0-5.5,6.0 可能撞硬件墙。
    """

    GATE = {"v_lo": 2.0, "v_hi": 5.5}
    GATE_MIN = 0.03

    def __post_init__(self):
        super().__post_init__()
        self.rewards.dof_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2_gated, weight=-2.0e-5,
            params={**self.GATE, "gate_min": self.GATE_MIN},
        )
        self.rewards.dof_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2_gated, weight=-2.5e-7,
            params={**self.GATE, "gate_min": self.GATE_MIN},
        )
        self.rewards.action_rate_l2 = RewTerm(
            func=mdp.action_rate_l2_gated, weight=-0.05,
            params={**self.GATE, "gate_min": self.GATE_MIN},
        )
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 6.0)


@configclass
class BpxProFlatSprint8EnvCfg_PLAY(BpxProFlatSprint8EnvCfg):
    """Sprint8 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint7EnvCfg_PLAY(BpxProFlatSprint7EnvCfg):
    """Sprint7 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint3EnvCfg_PLAY(BpxProFlatSprint3EnvCfg):
    """Sprint3 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprint2EnvCfg_PLAY(BpxProFlatSprint2EnvCfg):
    """Sprint2 的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatSprintEnvCfg_PLAY(BpxProFlatSprintEnvCfg):
    """冲刺版的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (4.0, 4.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatFastEnvCfg_PLAY(BpxProFlatFastEnvCfg):
    """高速版的回放 / 导出配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


@configclass
class BpxProFlatEnvCfg_PLAY(BpxProFlatEnvCfg):
    """play / 导出用：少量环境、不随机化、指令固定。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_com = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
