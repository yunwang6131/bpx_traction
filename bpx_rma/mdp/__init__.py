"""BPX-Pro 的 MDP 项。

IsaacLab 自带的 mdp 整个透出来，再覆盖上本项目特有的事件 / 观测，
这样环境配置里一个 `from bpx_rma import mdp` 就够了。
"""

# isaaclab_tasks 的 velocity mdp 已经 re-export 了 isaaclab.envs.mdp，
# 再加上 feet_air_time / feet_slide / terrain_levels_vel 这些走路专用项
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .events import knee_load_friction, randomize_actuator_params, randomize_total_mass  # noqa: F401
from .gait import gait_phase_obs  # noqa: F401
from .rewards import (  # noqa: F401
    action_rate_l2_gated,
    feet_gait,
    flat_orientation_roll_l2,
    foot_clearance_deficit,
    foot_clearance_deficit_gated,
    foot_clearance_penalty,
    joint_acc_l2_gated,
    joint_torques_l2_gated,
    lin_vel_deficit,
)
from .observations import (  # noqa: F401
    actuator_armature_ratio,
    actuator_friction_ratio,
    actuator_viscous_ratio,
    encoder_bias,
    feet_contact,
    feet_contact_force,
    joint_applied_torque,
    joint_vel_filtered,
    knee_friction_abs,
    knee_load,
    mass_offset,
    torque_delay_steps,
    total_mass,
)
from .state import BpxDomainState, get_state  # noqa: F401
