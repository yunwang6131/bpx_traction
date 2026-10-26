"""步态时钟（gait phase）—— 只驱动奖励，不进观测。

关键点：相位是环境里的一个纯时间函数，奖励函数直接算、直接用，**不拼进 45 维
观测**。所以板子契约完全不动，蒸馏/导出/上板流程一字不改；策略靠本体感觉历史
自学节拍，gait 奖励只是把这个自学的节律塑形得更规律。

腿序 = ``spec.FOOT_BODIES`` = [FL, FR, HL, HR]。对角 trot 的相位偏移
``[0.5, 0, 0, 0.5]``：FL&HR 同相、FR&HL 同相，两组差半周期。

固定周期版：``φ = (episode_length_buf × step_dt) mod T / T``。真机侧如果要做
变频（步频随速度），把 T 换成命令项提供的频率即可，这里先用固定 T。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .. import spec

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 默认对角 trot 偏移，顺序同 spec.FOOT_BODIES [FL, FR, HL, HR]
TROT_OFFSET: tuple[float, ...] = (0.5, 0.0, 0.0, 0.5)


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    """全局步态相位 φ ∈ [0, 1)，(N,)。"""
    return (env.episode_length_buf * env.step_dt) % period / period


def leg_phase(env: ManagerBasedRLEnv, period: float, offset: Sequence[float]) -> torch.Tensor:
    """每条腿的相位 (N, 4)，顺序同 spec.FOOT_BODIES。"""
    phi = gait_phase(env, period).unsqueeze(1)
    off = torch.as_tensor(offset, device=env.device, dtype=phi.dtype).view(1, -1)
    return torch.remainder(phi + off, 1.0)


def foot_z_world(env: ManagerBasedRLEnv) -> torch.Tensor:
    """四脚 toe_link 世界系 z，(N, 4)，顺序同 spec.FOOT_BODIES。

    平地触地时 ≈ FOOT_RADIUS 0.0285。body_ids 用 preserve_order 解析一次缓存，
    保证列顺序和 TROT_OFFSET 对齐（SceneEntityCfg 不保证顺序）。
    """
    if not hasattr(env, "_gait_foot_ids"):
        asset = env.scene["robot"]
        ids, _ = asset.find_bodies(spec.FOOT_BODIES, preserve_order=True)
        env._gait_foot_ids = torch.tensor(ids, device=env.device)
    return env.scene["robot"].data.body_pos_w[:, env._gait_foot_ids, 2]


def command_active(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """指令激活 mask (N,)：‖v_xy‖ + |yaw| > threshold。站立时 gait 奖励不激活。"""
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs() > threshold).float()


def gait_phase_obs(
    env: ManagerBasedRLEnv,
    period: float,
    command_name: str = "base_velocity",
    stand_threshold: float = 0.1,
) -> torch.Tensor:
    """相位观测 (N, 2) = [sin(2πφ), cos(2πφ)]，追加在 45 维帧的末尾。

    用 sin/cos 两维编码一个周期相位是周期量的标准做法（避免 φ 在 0/1 之间跳变），
    不是某个机器人专有的。时钟 φ 用的是 `gait_phase`——和 feet_gait 奖励**同一个**
    时钟（周期按我们自己步频定），保证"观测给的节拍"和"奖励逼的节拍"一致。

    站立（‖cmd‖ 低于阈值）时输出置零：不走就没有节拍。底层时钟仍在跑，一旦给出
    指令相位就接着用。板子侧将来复现时，这个时钟是纯时间函数、能精确算出来。
    """
    phi = gait_phase(env, period)                                 # (N,)
    out = torch.stack([torch.sin(2 * torch.pi * phi), torch.cos(2 * torch.pi * phi)], dim=1)
    stand = command_active(env, command_name, stand_threshold).unsqueeze(1)  # (N,1) 走=1 站=0
    return out * stand
