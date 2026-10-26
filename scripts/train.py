#!/usr/bin/env python3
"""BPX-Pro RMA 训练。两个阶段用同一个脚本，靠 `--agent` 切换。

阶段一 —— 特权 PPO，训出 teacher：

    conda activate bpx_rma
    python scripts/train.py --headless --num_envs 4096

阶段二 —— 蒸馏成只看历史的 student（= RMA，可以导出上真机）：

    python scripts/train.py --headless --num_envs 4096 \
        --agent rsl_rl_distillation_cfg_entry_point

阶段二会自动从 `logs/rsl_rl/bpx_pro_flat/` 下最新的 `*_teacher` run 里取 checkpoint。

对照实验（厂家参数 vs 辨识参数）换一份参数文件就行，日志目录也分开：

    BPX_PRO_PARAMS_YAML=data/bpx_pro_baseline_params.yaml \
        python scripts/train.py --headless --run-name teacher_baseline
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="BPX-Pro RMA 训练（PPO teacher / 蒸馏 student）")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-v0")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="rsl_rl_cfg_entry_point = 阶段一 PPO；rsl_rl_distillation_cfg_entry_point = 阶段二蒸馏",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--run-name", dest="run_name", type=str, default=None, help="覆盖日志目录后缀")
parser.add_argument(
    "--experiment-name", dest="experiment_name", type=str, default=None,
    help="覆盖实验名（= logs/rsl_rl/<名字>/）。对照实验必须换名字，"
         "否则蒸馏找 teacher 的正则 '.*_teacher' 会同时匹配到另一组的 run",
)
parser.add_argument("--resume", action="store_true", help="接着上一次同类型的 run 继续练")
parser.add_argument("--load-run", dest="load_run", type=str, default=None, help="指定 run 目录的正则")
parser.add_argument("--logger", type=str, default=None, choices=["tensorboard", "wandb", "neptune"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""以下必须在 Isaac Sim 起来之后 import。"""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401  —— import 即注册 gym 任务

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if args_cli.resume:
        agent_cfg.resume = True

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))

    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # 注意看 runner 的类名而不是算法的：算法被换成了自定义的 RmaDistillation
    is_distillation = agent_cfg.class_name == "DistillationRunner"

    log_root = (PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = str(log_root / log_dir)
    print(f"[bpx] 日志目录：{log_dir}")

    # 蒸馏要先找到 teacher 的 checkpoint（必须在新建 log_dir 之前找，否则会把空目录也算进去）
    resume_path = None
    if agent_cfg.resume or is_distillation:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[bpx] {'teacher' if is_distillation else '续训'} checkpoint：{resume_path}")

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = DistillationRunner if agent_cfg.class_name == "DistillationRunner" else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if resume_path is not None:
        # 迁移种子（跨观测维度，如相位进观测 45→47）没有 optimizer_state_dict：
        # 权重已按新维度扩好，只需加载 actor/critic，optimizer/iteration 从头重建。
        import torch as _torch

        _ck = _torch.load(resume_path, weights_only=False, map_location="cpu")
        _migrated = "optimizer_state_dict" not in _ck
        del _ck
        if _migrated:
            print("[bpx] 迁移种子：跳过 optimizer/iteration，从头重建")
            runner.load(resume_path, load_cfg={"actor": True, "critic": True, "optimizer": False,
                                               "iteration": False, "rnd": False}, strict=False)
        else:
            runner.load(resume_path)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # 把这次用的辨识参数一并存档，方便事后对照是哪一版
    from bpx_rma.params import get_params

    dump_yaml(os.path.join(log_dir, "params", "identification.yaml"), get_params().raw)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
