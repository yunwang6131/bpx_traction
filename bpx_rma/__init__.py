"""BPX-Pro 的 RMA 师生训练：用策略验证执行器辨识结果。

import 这个包会把 gym 任务注册上（需要 Isaac Sim 已经启动，所以要在
`AppLauncher` 之后 import）：

  ``Isaac-Velocity-Flat-BpxPro-v0``        训练用
  ``Isaac-Velocity-Flat-BpxPro-Play-v0``   回放 / 导出用

两个 agent 入口：
  ``rsl_rl_cfg_entry_point``               阶段一 PPO（teacher）
  ``rsl_rl_distillation_cfg_entry_point``  阶段二蒸馏（student = RMA）

注意：这里**不要** import `agents` 或 `velocity_env_cfg` —— 它们依赖 isaaclab，
而 `spec` / `params` 要能在没有 Isaac Sim 的环境里单独用（`sim2sim.py` 只跑 MuJoCo + ONNX）。
gym 的 entry point 是字符串、惰性解析的，所以写字面量就够了。
"""

import gymnasium as gym

from . import spec  # noqa: F401  纯 Python，无 isaaclab 依赖

_ENV = f"{__name__}.velocity_env_cfg"
_AGENTS = f"{__name__}.agents.rsl_rl_cfg"

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatFastEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprintEnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint7-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint7EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint9-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint9EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint8-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint8EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprint2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Sprint-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatSprintEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-BpxPro-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_ENV}:BpxProFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}:BpxProFlatPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": f"{_AGENTS}:BpxProFlatDistillationRunnerCfg",
    },
)
