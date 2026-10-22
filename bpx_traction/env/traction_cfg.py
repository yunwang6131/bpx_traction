"""把任意一个 BPX 任务的环境配置改造成"能采牵引数据"的版本。

**不注册新任务、不继承具体的 cfg 类、不碰 `bpx_rma`。** 写成一个就地改配置的函数，
理由是基底策略会换 —— 现在是 rollfix / sprint7，以后还会有别的。如果这里写死继承
`BpxProFlatEnvCfg`，每换一个基底就要加一个子类和一条 gym 注册；而采集本身对基底是
无所谓的，它只需要"这个环境里有 IMU、没有别的扰动"。

用法::

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    prepare_for_traction(env_cfg, num_envs=512)
    env = gym.make(task, cfg=env_cfg)
"""

from __future__ import annotations

from isaaclab.sensors import ImuCfg

from .. import spec

#: 场景里 Imu 传感器的名字，`FrameAssembler` 按这个名字找。
IMU_SENSOR_NAME = "imu"


def make_imu_cfg(body_name: str | None = None) -> ImuCfg:
    """机身 IMU。

    ``gravity_bias=(0, 0, 9.81)`` 是 IsaacLab 的默认值，**必须保留** —— 它让
    `lin_acc_b` 输出**比力**（含重力），和真 IMU 一致。设成 (0,0,0) 会得到纯运动学
    加速度，静止时读数是 0 而不是 9.81，训出来的估计器一上真机就废。
    `normalize.sanity_check()` 会盯着这一条。

    手册 2.3.3：「IMU 传感器固连在身体内部，其坐标系与身体坐标系平行（无旋转变换）」，
    所以 offset 用默认的单位姿态，不需要标安装旋转。
    """
    body = body_name or spec.TETHER_BODY      # torso
    return ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + body,
        gravity_bias=(0.0, 0.0, 9.81),
        debug_vis=False,
    )


def prepare_for_traction(
    env_cfg,
    num_envs: int | None = None,
    disable_push: bool = True,
    disable_corruption: bool = True,
    episode_length_s: float | None = None,
):
    """就地改配置，返回同一个对象。

    Args:
        env_cfg: `load_cfg_from_registry(task, "env_cfg_entry_point")` 的结果。
        num_envs: 并行环境数。
        disable_push: 关掉 `push_robot`。**采牵引数据时必须关** —— 那是瞬时速度扰动，
            它产生的本体响应和外力很像，但它**没有对应的力标签**，等于往数据里掺入
            一批"有响应、标签却是零力"的样本，直接教估计器低估。
        disable_corruption: 关掉观测噪声。噪声要按真机实测的幅度注入（见
            `scripts/measure_sensor_noise.py`），不用训练环境里那套。
        episode_length_s: 回合长度，默认按最长的力剖面留余量。
    """
    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs

    # 加 IMU。场景 cfg 是 configclass，直接赋属性即可。
    setattr(env_cfg.scene, IMU_SENSOR_NAME, make_imu_cfg())

    if disable_corruption:
        env_cfg.observations.policy.enable_corruption = False
        if hasattr(env_cfg.observations, "history"):
            env_cfg.observations.history.enable_corruption = False

    if disable_push:
        if getattr(env_cfg.events, "push_robot", None) is not None:
            env_cfg.events.push_robot = None
        # base_com 随机化保留 —— 那是域随机化，属于"同一台机器人的不确定性"，
        # 不会伪造出假的外力标签。

    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s

    return env_cfg


def summarize(env_cfg) -> str:
    imu = getattr(env_cfg.scene, IMU_SENSOR_NAME, None)
    lines = [
        f"环境       num_envs={env_cfg.scene.num_envs}  episode={env_cfg.episode_length_s:.1f}s",
        f"IMU        {'有' if imu else '没有'}"
        + (f"  gravity_bias={imu.gravity_bias}（含重力，必须）" if imu else ""),
        f"push_robot {'关' if getattr(env_cfg.events, 'push_robot', None) is None else '⚠️ 还开着'}",
        f"观测噪声   {'关' if not env_cfg.observations.policy.enable_corruption else '⚠️ 还开着'}",
        f"指令区间   vx={tuple(env_cfg.commands.base_velocity.ranges.lin_vel_x)}  "
        f"vy={tuple(env_cfg.commands.base_velocity.ranges.lin_vel_y)}",
    ]
    return "\n".join(lines)
