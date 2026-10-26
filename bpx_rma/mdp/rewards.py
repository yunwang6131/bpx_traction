"""BPX-Pro 的奖励项。

本版 IsaacLab 的通用 velocity mdp 没有足端离地高度奖励，只有 Spot 配置里有一份
（`...velocity/config/spot/mdp/rewards.py:foot_clearance_reward`）。这里按同一
口径抄一份，不去动 IsaacLab 源码树。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from . import gait as _gait

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# 速度门控的能量惩罚：低速全额（省能平稳），高速放松（让策略敢发力冲）
# ---------------------------------------------------------------------------
def _speed_gate(
    env: ManagerBasedRLEnv, command_name: str, v_lo: float, v_hi: float, gate_min: float
) -> torch.Tensor:
    """(N,) 门控系数：|cmd_vx| ≤ v_lo → 1；≥ v_hi → gate_min；之间线性。

    用**指令速度**而非实测速度（实测有噪声、且站桩时实测=0 会误判成低速反而重罚，
    把站桩锁死）。指令是策略当前被要求的速度，高指令时就该放松，鼓励它去冲。
    """
    v = env.command_manager.get_command(command_name)[:, 0].abs()
    frac = ((v - v_lo) / max(v_hi - v_lo, 1e-6)).clamp(0.0, 1.0)
    return 1.0 - (1.0 - gate_min) * frac


def joint_torques_l2_gated(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity",
    v_lo: float = 2.0, v_hi: float = 4.5, gate_min: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ τ²，乘速度门控。高速时放松，让策略敢用大力矩推进。"""
    asset: Articulation = env.scene[asset_cfg.name]
    base = torch.sum(torch.square(asset.data.applied_torque), dim=1)
    return base * _speed_gate(env, command_name, v_lo, v_hi, gate_min)


def joint_acc_l2_gated(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity",
    v_lo: float = 2.0, v_hi: float = 4.5, gate_min: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Σ q̈²，乘速度门控。高速时放松，让策略敢爆发式蹬腿。"""
    asset: Articulation = env.scene[asset_cfg.name]
    base = torch.sum(torch.square(asset.data.joint_acc), dim=1)
    return base * _speed_gate(env, command_name, v_lo, v_hi, gate_min)


def action_rate_l2_gated(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity",
    v_lo: float = 2.0, v_hi: float = 4.5, gate_min: float = 0.2,
) -> torch.Tensor:
    """Σ(a_t − a_{t−1})²，乘速度门控。高速时放松，让动作可以变快。"""
    base = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return base * _speed_gate(env, command_name, v_lo, v_hi, gate_min)


def foot_clearance_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, tanh_mult: float
) -> torch.Tensor:
    """摆动相抬脚高度惩罚：足端世界系 z 偏离 target_height 时扣分（权重取负）。

    误差按足端**水平速度**的 tanh 门控——支撑脚（水平速度 ~0）不计入，
    摆得越快权重越接近 1，所以它只塑形摆动相的抬腿高度，不干扰支撑。

    故意**不用** Spot 的 exp 正奖励形式：那个公式在脚完全不动时门控为 0、
    exp(0)=1 拿满分，sprint4 第一次训练就被策略钻了这个空子（集体站桩，
    跟踪奖励 1.2→0.44）。惩罚形式下站桩 = 0 分，中性，走不走由跟踪奖励决定。

    注意 target_height 是 toe_link 原点（碰撞圆柱中心）的世界 z：
    站立触地时 ≈ FOOT_RADIUS 0.0285 m，target 0.10 ≈ 离地 7 cm。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    return torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)


def flat_orientation_roll_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """只惩罚横滚(左右倾)的机身姿态项：proj_grav 的 y 分量平方。

    body 系 x 前 / y 左 / z 上，正立时 proj_grav=(0,0,-1)。右倾(右低)时重力偏向
    body -y，所以 grav_y 就是横滚指示，与俯仰(grav_x)解耦。

    背景：sprint 系列为了高速把 `flat_orientation_l2`(罚 grav_x²+grav_y²)从 -1.0
    放松到 -0.2，俯仰自由了，但横滚约束也跟着松，随机对称破缺长到 ~2°(flat/fast
    用 -1.0 时只有 <1°)。这一项单独把横滚压回去，不动俯仰的放松。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def foot_clearance_deficit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, tanh_mult: float
) -> torch.Tensor:
    """单边线性抬脚亏空惩罚：摆动相足端低于 target_height 的差值（权重取负）。

    和 :func:`foot_clearance_penalty`（二次型、对称）的两点差异，都是为了破
    "峰值卡在 2.5~3 cm" 的平台期（-2 → -6 权重三倍峰值不动）：

    * **单边**：高于目标零惩罚，不产生把弧线往下拉的对称项；
    * **线性**：平台区梯度恒定，而二次型在小误差区梯度趋零、在起脚/落脚
      瞬间（脚必然又低又快）的大误差不可避免项反而主导了总惩罚。
      线性亏空对起脚动作的梯度方向是"先抬脚、再前摆"，正是想要的形态。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    deficit = (target_height - asset.data.body_pos_w[:, asset_cfg.body_ids, 2]).clamp(min=0.0)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    return torch.sum(deficit * foot_velocity_tanh, dim=1)


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    duty: float,
    offset: Sequence[float] = _gait.TROT_OFFSET,
    contact_height: float = 0.045,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """接触-相位匹配（权重取正）：支撑相该触地、摆动相该离地，匹配比例。

    is_stance = leg_phase < duty；is_contact = 脚高 < contact_height（平地触地约
    0.0285）。四脚匹配比例 ∈ {0, 0.25, 0.5, 0.75, 1}，强制规律对角 trot。
    用脚高判接触（和 play 口径一致），避免接触传感器 body 顺序对不齐。
    """
    foot_z = _gait.foot_z_world(env)
    lp = _gait.leg_phase(env, period, offset)
    is_stance = lp < duty
    is_contact = foot_z < contact_height
    reward = (is_stance == is_contact).float().mean(dim=1)
    return reward * _gait.command_active(env, command_name, command_threshold)
