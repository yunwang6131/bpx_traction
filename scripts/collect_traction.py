#!/usr/bin/env python3
"""采力估计器的训练数据。

每一轮：给每个并行环境抽一条力剖面 + 一个速度指令，随机挂点，跑策略，
攒 25 帧历史，打 15 维标签。多个策略轮流跑，数据混在一起。

    conda activate bpx_rma
    python scripts/collect_traction.py --headless --num-envs 512 --rounds 8 \
        --policy rollfix=logs/rsl_rl/bpx_pro_sprint4/2026-10-21_11-02-37_student \
        --policy sprint7=logs/rsl_rl/bpx_pro_sprint7/2026-10-22_20-00-21_student

## 为什么要混多个策略

估计器的 61 维输入里有 `last_action`（12 维），它是**策略的输出**。换一个基底策略，
last_action 的分布就变了，估计器的输入分布跟着变。

物理映射没变 —— `target = q0 + 0.1875·raw → PD(kp=40, kd=1)` 是契约锁死的，
变的只是 last_action 的取值分布。所以这是分布偏移而不是根本性失效，
但步态差别大的策略（flat vs sprint）偏移会很明显。

混着采就让估计器对这件事不敏感，顺带对"以后再换基底"免疫。成本是零 ——
检查点都现成，只是采集时轮换加载。

**不要用 flat 采**：扛力扫描实测它站立 10 N 就开始摔
（`logs/traction/drag_sweep.json`），摔倒的轨迹采不到有用的外力响应。
用 rollfix / sprint7 这类扛得住的。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="采牵引力估计器的训练数据")
parser.add_argument("--policy", action="append", default=[], metavar="名字=路径",
                    help="采数据用的策略，可重复。别用 flat（扛不住力）")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")

parser.add_argument("--num-envs", dest="num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=8, help="每个策略跑几轮")
parser.add_argument("--steps-per-round", dest="steps_per_round", type=int, default=400)
parser.add_argument("--stride", type=int, default=5, help="每多少步出一个样本")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fixed-attach", dest="fixed_attach", action="store_true",
                    help="挂点固定在标称点（消融用：和随机挂点比力精度）")
parser.add_argument("--output", type=str, default=None,
                    help="输出目录，默认 logs/traction/dataset")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

if not args_cli.policy:
    parser.error("至少要给一个 --policy 名字=路径")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401  注册任务
from bpx_traction import spec  # noqa: E402
from bpx_traction.data import collect  # noqa: E402
from bpx_traction.env import traction_cfg  # noqa: E402
from bpx_traction.normalize import Normalizer, sanity_check  # noqa: E402


def load_agent_cfg():
    cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    return handle_deprecated_rsl_rl_cfg(cfg, _pkg_version("rsl-rl-lib"))


def resolve_checkpoint(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到策略：{path}")
    return Path(get_checkpoint_path(str(path.parent), path.name, args_cli.checkpoint))


def main() -> None:
    policies = []
    for item in args_cli.policy:
        if "=" not in item:
            raise ValueError(f"--policy 要写成 名字=路径，拿到 {item!r}")
        name, _, raw = item.partition("=")
        policies.append((name.strip(), resolve_checkpoint(raw.strip())))

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    traction_cfg.prepare_for_traction(
        env_cfg,
        num_envs=args_cli.num_envs,
        episode_length_s=max(60.0, 4 * args_cli.steps_per_round * step_dt),
    )
    print("[traction] 采集环境：\n" + traction_cfg.summarize(env_cfg))

    env = RslRlVecEnvWrapper(
        gym.make(args_cli.task, cfg=env_cfg), clip_actions=load_agent_cfg().clip_actions
    )

    cfg = collect.CollectionConfig(
        steps_per_round=args_cli.steps_per_round,
        stride=args_cli.stride,
        rounds=args_cli.rounds,
        randomize_attach=not args_cli.fixed_attach,
        seed=args_cli.seed,
    )
    collector = collect.TractionCollector(env, cfg)
    print("[traction] " + collector.frames.describe())
    print("[traction] " + collector.tether.describe())

    # 第一帧就核对：估计器帧和策略帧的公共项必须逐位相同。
    # 关节顺序接错、默认位姿减错、last_action 取了不同时刻 —— 这几类错不会让训练崩，
    # 只会让估计器学到一个偏的映射，上真机才暴露。在这里挡掉。
    obs = env.get_observations()
    collector.frames.assert_matches_policy_frame(obs["policy"])
    print("[traction] ✓ 估计器帧与策略帧的 5 组公共量逐位一致")

    rollouts = []
    for policy_id, (name, checkpoint) in enumerate(policies):
        agent_cfg = load_agent_cfg()
        is_distill = agent_cfg.class_name == "DistillationRunner"
        runner_cls = DistillationRunner if is_distill else OnPolicyRunner
        runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if is_distill:
            runner.load(str(checkpoint),
                        load_cfg={"student": True, "teacher": True, "iteration": False})
        else:
            runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=agent_cfg.device)

        for rnd in range(args_cli.rounds):
            data = collector.run_round(policy, policy_id=policy_id)
            rollouts.append(data)
            print(f"[traction] {name} 第 {rnd + 1}/{args_cli.rounds} 轮  +{len(data):,} 样本")

    env.close()

    merged = collect.concat(rollouts)
    print("\n[traction] 合并后：\n" + collect.summarize(merged))

    masks = collect.split_by_env(merged, args_cli.num_envs)
    out = Path(args_cli.output) if args_cli.output else spec.TRACTION_LOG_DIR / "dataset"
    out.mkdir(parents=True, exist_ok=True)

    sizes = {}
    for split, mask in masks.items():
        np.savez_compressed(
            out / f"{split}.npz",
            history=merged.history[mask],
            targets=merged.targets[mask],
            env_id=merged.env_id[mask],
            policy_id=merged.policy_id[mask],
            attach_offset=merged.attach_offset[mask],
            profile_kind=merged.profile_kind[mask],
            since_change=merged.since_change[mask],
        )
        sizes[split] = int(mask.sum())
        print(f"[traction] {split:<11s} {sizes[split]:>8,}  → {out / (split + '.npz')}")

    # 归一化只从训练集统计 —— 用全量统计会把验证/测试集的信息泄进来
    normalizer = Normalizer.from_history(merged.history[masks["train"]])
    normalizer.save(out / "normalization.npz")
    warnings = sanity_check(normalizer)
    if warnings:
        print("\n[traction] ⚠️ 归一化自检有话说：")
        for w in warnings:
            print("   " + w)
    else:
        print("[traction] ✓ 归一化自检无异常（含重力的加速度、归一的投影重力、四元数）")

    manifest = {
        "task": args_cli.task,
        "policies": {name: str(path) for name, path in policies},
        "num_envs": args_cli.num_envs,
        "rounds_per_policy": args_cli.rounds,
        "steps_per_round": args_cli.steps_per_round,
        "stride": args_cli.stride,
        "seed": args_cli.seed,
        "frame": {
            "size": spec.FRAME_SIZE,
            "history_length": spec.HISTORY_LENGTH,
            "newest_last": spec.HISTORY_NEWEST_LAST,
            "terms": [{"name": n, "dim": d} for n, d in spec.TERMS],
        },
        "targets": [{"name": n, "dim": d} for n, d in spec.OUTPUT_TERMS],
        "label_frame": spec.FORCE_FRAME,
        "attach": {
            "randomized": not args_cli.fixed_attach,
            "body": spec.TETHER_BODY,
            "nominal": list(spec.TETHER_OFFSET_B),
            "range": [list(r) for r in spec.TETHER_OFFSET_RANGE_B],
            "confirmed_on_hardware": spec.TETHER_OFFSET_CONFIRMED,
        },
        "magnitudes_n": list(cfg.magnitudes_n),
        "profile_weights": cfg.profile_weights,
        "split": "按 env 索引 70/15/15；相邻窗口重叠 24/25 帧，按样本随机切会泄漏",
        "sizes": sizes,
        "push_robot_disabled": True,
        "normalization_from": "train split only",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"[traction] manifest → {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
