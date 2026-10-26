"""BPX-Pro 的部署契约与机器人规格。

这里的每个常数都有唯一来源，改之前先确认来源那边一致：

* **部署契约**：控制板 `<board-host>:~/wy/bpx_flat_freq_np3o_deploy_v4/bpx_dual_pad.py`
  （内网主机，真实地址见本地部署记录；存档在 `pace/robot_cat/bpx_dual_pad_flush.py`）。板子上的
  `make_policy_session()` 硬性要求 ONNX 是 2 输入 1 输出，`POLICY_TERM_SLICES` /
  `ACTION_SCALE` / `KP_POLICY` 全是写死的。训练必须对齐，否则导出的策略上不了机。
* **机器人结构**：mirrormerobotics/BPX-Pro 的 `bpxPro.urdf` / `usd/bpxPro.usd`。
* **辨识参数**：`data/bpx_pro_training_params.yaml`（由 :mod:`bpx_rma.params` 读取）。

本模块是纯 Python，不 import isaaclab，所以导出脚本 / 离线检查不启 Isaac Sim 也能用。
"""

from __future__ import annotations

import os
from pathlib import Path

ROBOT_NAME = "bpx_pro"

# ---------------------------------------------------------------------------
# 关节顺序：bpx_sdk JointIndex 0..11 = URDF 关节名 = 策略 / MJCF 顺序
# 板子上的注释："BPX velocity-RMA policy/MJCF order: FL, FR, HL, HR x roll, pitch, knee"
# ---------------------------------------------------------------------------
LEG_NAMES = ("fl", "fr", "hl", "hr")
JOINT_TYPES = ("hip_roll", "hip_pitch", "knee")
JOINT_NAMES: list[str] = [f"{leg}_{jt}_joint" for leg in LEG_NAMES for jt in JOINT_TYPES]
NUM_JOINTS = len(JOINT_NAMES)
assert NUM_JOINTS == 12

KNEE_JOINT_NAMES: list[str] = [f"{leg}_knee_joint" for leg in LEG_NAMES]
KNEE_INDICES: list[int] = [JOINT_NAMES.index(n) for n in KNEE_JOINT_NAMES]  # 2, 5, 8, 11

# 连杆名（URDF）
TORSO_BODY = "torso"
FOOT_BODIES: list[str] = [f"{leg}_toe_link" for leg in LEG_NAMES]
THIGH_BODIES: list[str] = [f"{leg}_thigh_link" for leg in LEG_NAMES]
CALF_BODIES: list[str] = [f"{leg}_calf_link" for leg in LEG_NAMES]

# ---------------------------------------------------------------------------
# 部署契约（板子上写死的，不能改）
# ---------------------------------------------------------------------------
#: 策略的默认关节角 q0。板子上 `DEFAULT_POSE`，同时也是 PD 起立的目标姿态。
DEFAULT_JOINT_POS: list[float] = [0.0, 0.7, -1.4] * 4

#: 板子上 `MODEL_LIMITS`，用于裁剪关节目标。注意 hip_pitch 上限用的是 URDF 的 2.60661。
JOINT_POS_LIMITS: list[tuple[float, float]] = [
    (-0.8727, 0.8727),   # hip_roll
    (-0.7854, 2.60661),  # hip_pitch
    (-2.7531, -0.5539),  # knee
] * 4

KP_POLICY = 40.0        # 板子上 KP_POLICY
KD_POLICY = 1.0         # 板子上 KD_POLICY
ACTION_SCALE = 0.1875   # 板子上 ACTION_SCALE：target = q0 + 0.1875 * raw
ACTION_CLIP = 4.0       # 板子上 ACTION_CLIP：raw = clip(网络输出, ±4)，导出的图里也带了这个 clamp

POLICY_HZ = 50.0        # 板子上 HZ_POLICY
JOINT_HZ = 500.0        # 板子上 HZ_JOINT（PD 内环）
SIM_DT = 1.0 / JOINT_HZ            # 0.002 s
DECIMATION = int(round(JOINT_HZ / POLICY_HZ))  # 10

#: 观测帧的分段（板子上 `POLICY_TERM_SLICES`）。顺序即拼接顺序，不能换。
OBS_TERMS: list[tuple[str, int]] = [
    ("joint_pos", 12),          # q - DEFAULT_JOINT_POS
    ("joint_vel", 12),          # dq
    ("base_ang_vel", 3),        # IMU 角速度（机身系）
    ("projected_gravity", 3),   # R(body->world)^T · [0,0,-1]
    ("last_action", 12),        # 上一步的 raw action（已 clip 到 ±4）
    ("velocity_command", 3),    # vx, vy, wz
]
OBS_FRAME_SIZE = sum(dim for _, dim in OBS_TERMS)      # 45
assert OBS_FRAME_SIZE == 45

#: 板子上 `POLICY_HISTORY_LENGTH`：adaptation module 吃的历史帧数。
HISTORY_LENGTH = 25
#: 板子上 `POLICY_HISTORY_INPUT_SIZE` = 45 * 25。
HISTORY_INPUT_SIZE = OBS_FRAME_SIZE * HISTORY_LENGTH   # 1125

#: ONNX 的输入 / 输出名，板子上 `make_policy_session()` 逐个校验。
ONNX_INPUT_POLICY = "policy"
ONNX_INPUT_HISTORY = "adaptation_history"
ONNX_OUTPUT = "actions"


def obs_term_slices() -> list[tuple[int, int]]:
    """单帧内每个分段的 [起, 止)，对应板子上的 `POLICY_TERM_SLICES`。"""
    out, begin = [], 0
    for _, dim in OBS_TERMS:
        out.append((begin, begin + dim))
        begin += dim
    return out


assert obs_term_slices() == [(0, 12), (12, 24), (24, 27), (27, 30), (30, 42), (42, 45)], (
    "单帧分段和板子上的 POLICY_TERM_SLICES 对不上"
)

# ---------------------------------------------------------------------------
# URDF 的力矩 / 速度上限（按关节类型）
# ---------------------------------------------------------------------------
# 力矩：官方《BLACK PANTHER X 二次开发手册》V1.0.3 的 9.1 身体参数写的是
# "最大关节扭矩 约等于 33 N·m"，没有区分关节。URDF / 官方 MJCF 的 ctrlrange 给膝 50 N·m，
# 两者冲突，按用户决定以手册为准。
# 影响：膝堵转力矩从 50×2=100 降到 33×2=66，高速段可用力矩少三分之一
#   （11.8 rad/s 时 21.3 -> 14.1 N·m）。低速段不受影响 —— 那里是 30 N·m 的部署限幅先生效。
EFFORT_LIMIT = {"hip_roll": 33.0, "hip_pitch": 33.0, "knee": 33.0}

# 速度：手册**没有**公布关节转速。
#
# ⚠️ 这里的 30 rad/s 是**待实测的假设值**（2026-10-25 按用户判断设定，计划下周用 SDK 的
# `getMaxVelocity()` 实测确认，手册 6.3）。实测完把三项都换成真值，并重测极速。
#
# 换掉的旧值是 URDF 的 limit 字段 `{hip: 22, knee: 15}`。为什么它不可能对：DCMotor 的
# 力矩-速度曲线在 `velocity_limit` 处可用力矩归零，而实测（logs/sim2sim/summary.json）
# 峰值膝速 / 体速 ≈ 4.6 rad/s 每 m/s，照 15 算零力矩速度只有 3.3 m/s —— 可 sprint7
# 实测跑到了 4.5。要么 URDF 的 15 偏小，要么 sprint7 的步态把这个比值压到 3.3 以下了
# （两者可能都成立）。
#
# 三项一起抬，不是只抬膝：髋留在 22 的话零力矩速度 22/4.26 = 5.2 m/s，髋会先撞墙，
# 6 m/s 照样无解。实测时三种关节类型都要问一遍，它们的减速比不一定相同。
VELOCITY_LIMIT = {"hip_roll": 30.0, "hip_pitch": 30.0, "knee": 30.0}

#: DCMotor 堵转力矩 = 力矩上限 × 该系数。参数 yaml 里也有同名项，
#: 这里留一份给不读 yaml 的脚本（如 sim2sim 覆盖 FloorSim 的力矩包络）用。
SATURATION_FACTOR_FALLBACK = 2.0

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent     # .../bpx/bpx_rma
PROJECT_DIR = PACKAGE_DIR.parent                  # .../bpx
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_PARAMS_YAML = DATA_DIR / "bpx_pro_training_params.yaml"
LOG_DIR = PROJECT_DIR / "logs"


def find_asset_dir() -> Path:
    """定位 mirrormerobotics/BPX-Pro 仓库（含 usd/bpxPro.usd）。

    顺序：环境变量 BPX_PRO_ASSET_DIR → 本项目同级 BPX-Pro → ~/intern/ZGYD/pace/BPX-Pro → ~/BPX-Pro
    """
    candidates: list[Path] = []
    if os.environ.get("BPX_PRO_ASSET_DIR"):
        candidates.append(Path(os.environ["BPX_PRO_ASSET_DIR"]).expanduser())
    candidates += [
        PROJECT_DIR.parent / "BPX-Pro",
        Path.home() / "intern" / "ZGYD" / "pace" / "BPX-Pro",
        Path.home() / "BPX-Pro",
    ]
    for c in candidates:
        if (c / "usd" / "bpxPro.usd").exists():
            return c.resolve()
    raise FileNotFoundError(
        "找不到 BPX-Pro 资产目录（需要 usd/bpxPro.usd）。"
        "设置环境变量 BPX_PRO_ASSET_DIR，或把仓库放到 " + str(PROJECT_DIR.parent / "BPX-Pro")
    )


def default_usd_path() -> str:
    """USD 路径：环境变量 BPX_PRO_USD_PATH 优先。"""
    if os.environ.get("BPX_PRO_USD_PATH"):
        return str(Path(os.environ["BPX_PRO_USD_PATH"]).expanduser().resolve())
    return str(find_asset_dir() / "usd" / "bpxPro.usd")


def joint_type(joint_name: str) -> str:
    """`fl_hip_roll_joint` -> `hip_roll`。"""
    for jt in ("hip_roll", "hip_pitch", "knee"):   # hip_roll 要在 hip_pitch 之前判断无所谓，互不前缀
        if joint_name.endswith(f"_{jt}_joint"):
            return jt
    raise ValueError(f"认不出的关节名：{joint_name}")


def per_joint(by_type: dict[str, float]) -> list[float]:
    """把按关节类型给的值展开成 12 维（SDK 顺序）。"""
    return [by_type[joint_type(n)] for n in JOINT_NAMES]
