#!/usr/bin/env python3
"""扛力扫描：现有策略在**持续外力**下，到多少牛开始被拖走。

这是牵引这条线的**裁决实验**，而且它不需要力估计器、不需要数据、不需要训练 ——
只要一个已经训好的策略。

## 它回答什么

"力映射成速度指令"这条路成立的前提是：**我发什么速度，机器人就走什么速度**。
力小的时候策略能压住扰动，实际速度 = 指令速度，链路成立。力大到策略压不住，
机器人被外力**直接拖着动**，实际速度 ≠ 指令速度 —— 这时候映射发什么都没意义了。

那个分界点在哪，决定了：

* 速度指令这条窄通道够不够用（够 → 收工，底层零改动）；
* 还是必须把力送进策略观测重训（45→49 / 47→51，板子契约要改、辨识对照线要分叉）。

在这个数出来之前，上面那个选择是在赌。

## 它怎么测

每个并行环境分到一格 (力大小 × 方向 × 速度指令)，跑一遍固定协议：

    settle(无力，让步态稳下来) → ramp(加载) → hold(保持，在这一段测)

核心指标是 **drag** —— 跟踪误差在力方向上的投影：

    drag = (v_实际 − v_指令) · f̂        [机身系，只看水平分量]

策略完全压住外力时 drag ≈ 0；被拖走时 drag 为正且随力增大。
每个 (方向, 指令) 组合报一个**饱和力** = 最小的、drag 超阈值或直接摔倒的力。

## 怎么跑

    conda activate bpx_rma
    python scripts/drag_sweep.py --headless \
        --policy rollfix=logs/rsl_rl/bpx_pro_sprint4/2026-10-21_11-02-37_student \
        --policy flat=logs/rsl_rl/bpx_pro_flat/2026-10-10_11-42-30_student

多个 `--policy 名字=路径` 会在同一个 Isaac 会话里依次跑，共用同一套力场景 ——
所以策略之间的比较是同条件的。路径给 run 目录或直接给 .pt。

**顺带对选基底有用**：现在 `play.py --report` 看的是跟踪误差、高度、膝力矩、步态对称性，
"持续外力下能扛多少"是个还没有的维度。Sprint7 放开了力矩/加速度惩罚去冲极速，
很可能同时改变了抗外力能力（能用大力矩了顶得更住，但姿态平稳度下降更容易被推倒）。
扫一遍就知道。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="BPX-Pro 持续外力扛力扫描")
parser.add_argument(
    "--policy", action="append", default=[], metavar="名字=路径",
    help="要扫的策略，可重复。路径给 run 目录或 .pt 检查点",
)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt",
                    help="run 目录下挑检查点的正则")

parser.add_argument("--magnitudes", type=float, nargs="+",
                    default=[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
                    help="力的档位 (N)")
parser.add_argument("--azimuths", type=float, nargs="+", default=None,
                    help="水平方位角 (度)，默认 8 向。0=正前 90=正左")
parser.add_argument("--speeds", type=float, nargs="+", default=[0.0, 0.5, 1.0],
                    help="前进速度指令档位 (m/s)")

parser.add_argument("--settle-s", dest="settle_s", type=float, default=1.0)
parser.add_argument("--ramp-s", dest="ramp_s", type=float, default=0.5)
parser.add_argument("--hold-s", dest="hold_s", type=float, default=3.0)
parser.add_argument("--measure-last-s", dest="measure_last_s", type=float, default=2.0,
                    help="在 hold 的最后这么多秒里取指标，避开加载瞬态")

parser.add_argument("--drag-threshold", dest="drag_threshold", type=float, default=0.3,
                    help="drag 超过它判为'被拖走' (m/s)")
parser.add_argument("--tilt-limit", dest="tilt_limit", type=float, default=45.0,
                    help="|roll| 或 |pitch| 超过它判为失败 (度)")
parser.add_argument("--height-limit", dest="height_limit", type=float, default=0.20,
                    help="机身高度低于它判为失败 (m)。BPX 正常站 0.32~0.40")

parser.add_argument("--repeats", type=int, default=1, help="每格重复几份环境（取平均）")
parser.add_argument("--output", type=str, default=None,
                    help="报告落盘路径，默认 logs/traction/drag_sweep.json")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

if not args_cli.policy:
    parser.error("至少要给一个 --policy 名字=路径")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402


def load_agent_cfg():
    """rsl-rl 5.x 的 model cfg 换了 API，必须过一道 handle_deprecated_rsl_rl_cfg，
    否则 `MLPModel.__init__() got an unexpected keyword argument 'stochastic'`。
    和 play.py / train.py 走同一条路。"""
    cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    return handle_deprecated_rsl_rl_cfg(cfg, _pkg_version("rsl-rl-lib"))

import bpx_rma  # noqa: E402, F401  注册任务
from bpx_rma import spec as rma_spec  # noqa: E402
from bpx_traction import spec  # noqa: E402
from bpx_traction.data import scenarios  # noqa: E402
from bpx_traction.env.tether import TetherApplier  # noqa: E402


# ---------------------------------------------------------------------------
def resolve_checkpoint(path_str: str) -> Path:
    """`--policy 名字=路径` 里的路径：.pt 直接用，目录就按正则挑最新的。"""
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"找不到策略：{path}")
    return Path(get_checkpoint_path(str(path.parent), path.name, args_cli.checkpoint))


def parse_policies(items: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"--policy 要写成 名字=路径，拿到 {item!r}")
        name, _, raw = item.partition("=")
        out.append((name.strip(), resolve_checkpoint(raw.strip())))
    return out


def quat_to_roll_pitch_deg(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    w, x, y, z = quat_wxyz.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    return torch.rad2deg(roll), torch.rad2deg(pitch)


# ---------------------------------------------------------------------------
class SweepRunner:
    """一个 Isaac 会话里把所有策略依次扫一遍，场景完全相同。"""

    def __init__(self, conditions: list[scenarios.Condition], repeats: int):
        self.conditions = conditions
        self.repeats = repeats
        self.num_envs = len(conditions) * repeats

        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        env_cfg.scene.num_envs = self.num_envs
        env_cfg.observations.policy.enable_corruption = False
        # 扫的是**外力**，别的扰动必须关掉，否则分不清是谁把它推倒的
        env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "base_com"):
            env_cfg.events.base_com = None
        # 协议 4.5 s，留足余量不让它中途自动复位
        env_cfg.episode_length_s = max(60.0, 4 * self.protocol_duration_s())
        self.env_cfg = env_cfg

        self.env = RslRlVecEnvWrapper(
            gym.make(args_cli.task, cfg=env_cfg),
            clip_actions=load_agent_cfg().clip_actions,
        )
        self.base = self.env.unwrapped
        self.robot = self.base.scene["robot"]
        self.device = self.base.device
        self.step_dt = self.base.step_dt

        self.tether = TetherApplier(self.robot)
        print(f"[traction] {self.tether.describe()}")

        self.knee_ids, _ = self.robot.find_joints(rma_spec.KNEE_JOINT_NAMES, preserve_order=True)

        # 每个 env 的条件：按格铺开，repeats 份
        idx = np.tile(np.arange(len(conditions)), repeats)
        self.env_condition = torch.tensor(idx, dtype=torch.long, device=self.device)
        self.force_w = torch.tensor(
            np.stack([conditions[i].force.vector_w() for i in idx]),
            dtype=torch.float32, device=self.device,
        )                                                   # (N, 3)
        self.command = torch.tensor(
            np.stack([conditions[i].command.vector() for i in idx]),
            dtype=torch.float32, device=self.device,
        )                                                   # (N, 3)

    def protocol_duration_s(self) -> float:
        return args_cli.settle_s + args_cli.ramp_s + args_cli.hold_s

    # ----------------------------------------------------------------------
    def load_policy(self, checkpoint: Path):
        agent_cfg = load_agent_cfg()
        is_distill = agent_cfg.class_name == "DistillationRunner"
        runner_cls = DistillationRunner if is_distill else OnPolicyRunner
        runner = runner_cls(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        if is_distill:
            runner.load(str(checkpoint), load_cfg={"student": True, "teacher": True, "iteration": False})
        else:
            runner.load(str(checkpoint))
        return runner.get_inference_policy(device=agent_cfg.device)

    def run_one(self, policy) -> dict[str, np.ndarray]:
        """跑一遍协议，返回每个 env 的指标。"""
        n = self.num_envs
        settle = int(round(args_cli.settle_s / self.step_dt))
        ramp = int(round(args_cli.ramp_s / self.step_dt))
        hold = int(round(args_cli.hold_s / self.step_dt))
        measure_from = settle + ramp + hold - int(round(args_cli.measure_last_s / self.step_dt))
        total = settle + ramp + hold

        self.tether.clear()
        self.base.reset()
        obs = self.env.get_observations()

        alive = torch.ones(n, dtype=torch.bool, device=self.device)      # 首次终止后锁死
        ever_terminated = torch.zeros(n, dtype=torch.bool, device=self.device)
        drag_sum = torch.zeros(n, device=self.device)
        err_sum = torch.zeros(n, device=self.device)
        samples = torch.zeros(n, device=self.device)
        roll_peak = torch.zeros(n, device=self.device)
        pitch_peak = torch.zeros(n, device=self.device)
        min_height = torch.full((n,), 1e9, device=self.device)
        knee_tau = []

        force_xy = self.force_w[:, :2]
        force_mag = torch.linalg.norm(force_xy, dim=1)
        has_force = force_mag > 1e-6

        # 用 no_grad 而**不是** inference_mode：inference_mode 会把 IsaacLab 内部的
        # root_link_pose_w 之类的缓冲变成 inference tensor，下一个策略 reset() 在推理块
        # 外面写它就直接 RuntimeError。多策略依次扫必然踩到，play.py 只跑一个所以没事。
        with torch.no_grad():
            for step in range(total):
                # 指令每步都写 —— command manager 会按 resampling_time_range 覆盖
                self.base.command_manager.get_command("base_velocity")[:] = self.command

                # 力：settle 段为零，ramp 段线性加载，hold 段保持
                if step < settle:
                    scale = 0.0
                elif step < settle + ramp:
                    scale = (step - settle + 1) / max(ramp, 1)
                else:
                    scale = 1.0
                self.tether.apply_world(self.force_w * scale)

                action = policy(obs)
                obs, _, dones, _ = self.env.step(action)

                ever_terminated |= dones.bool()
                alive &= ~dones.bool()

                data = self.robot.data
                roll, pitch = quat_to_roll_pitch_deg(data.root_quat_w)
                live = alive.float()
                roll_peak = torch.maximum(roll_peak, roll.abs() * live)
                pitch_peak = torch.maximum(pitch_peak, pitch.abs() * live)
                min_height = torch.where(alive, torch.minimum(min_height, data.root_pos_w[:, 2]), min_height)

                if step >= measure_from:
                    # drag = 跟踪误差在力方向上的投影（机身系，只看水平）
                    v_err = data.root_lin_vel_b[:, :2] - self.command[:, :2]
                    f_b = self.tether.last_force_b[:, :2]
                    f_norm = torch.linalg.norm(f_b, dim=1, keepdim=True).clamp_min(1e-6)
                    drag = (v_err * (f_b / f_norm)).sum(dim=1)
                    drag_sum += torch.where(alive, drag, torch.zeros_like(drag))
                    err_sum += torch.where(alive, torch.linalg.norm(v_err, dim=1), torch.zeros_like(drag))
                    samples += alive.float()
                    knee_tau.append(data.applied_torque[:, self.knee_ids].abs().max(dim=1).values)

        self.tether.clear()
        counts = samples.clamp_min(1.0)
        tau = torch.stack(knee_tau) if knee_tau else torch.zeros(1, n, device=self.device)
        return {
            "drag_mps": (drag_sum / counts).cpu().numpy(),
            "lin_err_mps": (err_sum / counts).cpu().numpy(),
            "roll_peak_deg": roll_peak.cpu().numpy(),
            "pitch_peak_deg": pitch_peak.cpu().numpy(),
            "min_height_m": torch.where(min_height > 1e8, torch.zeros_like(min_height), min_height).cpu().numpy(),
            "knee_tau_peak_nm": tau.max(dim=0).values.cpu().numpy(),
            "terminated": ever_terminated.cpu().numpy(),
            "measured_steps": samples.cpu().numpy(),
            "has_force": has_force.cpu().numpy(),
        }

    def close(self):
        self.env.close()


# ---------------------------------------------------------------------------
def aggregate(conditions, per_env: dict[str, np.ndarray], repeats: int) -> list[dict]:
    """把 repeats 份环境按条件聚合。"""
    rows = []
    for i, cond in enumerate(conditions):
        sel = np.arange(len(conditions) * repeats) % len(conditions) == i
        failed = bool(
            per_env["terminated"][sel].any()
            or max(per_env["roll_peak_deg"][sel].max(), per_env["pitch_peak_deg"][sel].max())
            > args_cli.tilt_limit
            or per_env["min_height_m"][sel].min() < args_cli.height_limit
        )
        rows.append({
            "label": cond.label(),
            "magnitude_n": cond.force.magnitude_n,
            "azimuth_deg": cond.force.azimuth_deg,
            "command": cond.command.label(),
            "drag_mps": float(per_env["drag_mps"][sel].mean()),
            "lin_err_mps": float(per_env["lin_err_mps"][sel].mean()),
            "roll_peak_deg": float(per_env["roll_peak_deg"][sel].max()),
            "pitch_peak_deg": float(per_env["pitch_peak_deg"][sel].max()),
            "min_height_m": float(per_env["min_height_m"][sel].min()),
            "knee_tau_peak_nm": float(per_env["knee_tau_peak_nm"][sel].max()),
            "terminated": bool(per_env["terminated"][sel].any()),
            "failed": failed,
            "dragged": bool(per_env["drag_mps"][sel].mean() > args_cli.drag_threshold),
        })
    return rows


def saturation_forces(rows: list[dict]) -> dict[str, float | None]:
    """每个 (方向, 指令) 组合的饱和力 = 最小的"被拖走或摔倒"的力。None = 全程扛住。"""
    out: dict[str, float | None] = {}
    keys = sorted({(r["azimuth_deg"], r["command"]) for r in rows if r["magnitude_n"] > 0})
    for az, cmd in keys:
        group = sorted(
            (r for r in rows if r["azimuth_deg"] == az and r["command"] == cmd and r["magnitude_n"] > 0),
            key=lambda r: r["magnitude_n"],
        )
        hit = next((r["magnitude_n"] for r in group if r["failed"] or r["dragged"]), None)
        out[f"az{az:.0f}|{cmd}"] = hit
    return out


def render_table(name: str, rows: list[dict], saturation: dict) -> str:
    lines = [f"\n{'=' * 78}", f"策略 {name}", "=" * 78]
    commands = sorted({r["command"] for r in rows})
    magnitudes = sorted({r["magnitude_n"] for r in rows})

    for cmd in commands:
        lines.append(f"\n指令 {cmd}")
        lines.append(f"  {'力(N)':>6}  " + "  ".join(f"{'az' + str(int(a)):>7}" for a in sorted(
            {r['azimuth_deg'] for r in rows if r['magnitude_n'] > 0})))
        for mag in magnitudes:
            if mag == 0:
                base = [r for r in rows if r["magnitude_n"] == 0 and r["command"] == cmd]
                if base:
                    lines.append(f"  {mag:>6.0f}  无力基线 drag={base[0]['drag_mps']:+.3f} "
                                 f"lin_err={base[0]['lin_err_mps']:.3f} m/s")
                continue
            cells = []
            for az in sorted({r["azimuth_deg"] for r in rows if r["magnitude_n"] > 0}):
                r = next((x for x in rows if x["magnitude_n"] == mag
                          and x["azimuth_deg"] == az and x["command"] == cmd), None)
                if r is None:
                    cells.append(f"{'—':>7}")
                elif r["failed"]:
                    cells.append(f"{'摔':>7}")
                else:
                    cells.append(f"{r['drag_mps']:>+7.3f}")
            lines.append(f"  {mag:>6.0f}  " + "  ".join(cells))

    lines.append("\n饱和力（最小的被拖走/摔倒的力，— = 全程扛住）：")
    for key, value in sorted(saturation.items()):
        lines.append(f"  {key:<28s} {'—' if value is None else f'{value:.0f} N'}")

    held = [v for v in saturation.values() if v is None]
    hit = [v for v in saturation.values() if v is not None]
    max_tested = max((r["magnitude_n"] for r in rows), default=0.0)
    if hit:
        lines.append(
            f"\n  最弱方向 {min(hit):.0f} N   最强方向 {max(hit):.0f} N   "
            f"全程扛住的组合 {len(held)}/{len(saturation)}"
        )
    else:
        lines.append(
            f"\n  全部 {len(saturation)} 个组合在 {max_tested:.0f} N 以内都扛住了 —— "
            f"把 --magnitudes 往上加再扫一遍才能定出饱和点"
        )
    return "\n".join(lines)


def main() -> None:
    policies = parse_policies(args_cli.policy)
    azimuths = tuple(args_cli.azimuths) if args_cli.azimuths else scenarios.AZIMUTH_8
    commands = [scenarios.CommandSpec(vx=s) for s in args_cli.speeds]
    conditions = scenarios.condition_grid(
        scenarios.force_grid(args_cli.magnitudes, azimuths), commands
    )

    print(f"[traction] {scenarios.describe(conditions)}")
    print(f"[traction] 并行环境 {len(conditions) * args_cli.repeats}"
          f"（{len(conditions)} 格 × {args_cli.repeats} 份）")
    print(f"[traction] 协议 settle {args_cli.settle_s}s → ramp {args_cli.ramp_s}s "
          f"→ hold {args_cli.hold_s}s，在最后 {args_cli.measure_last_s}s 取指标")

    sweep = SweepRunner(conditions, args_cli.repeats)
    report = {
        "task": args_cli.task,
        "grid": scenarios.describe(conditions),
        "protocol": {
            "settle_s": args_cli.settle_s, "ramp_s": args_cli.ramp_s,
            "hold_s": args_cli.hold_s, "measure_last_s": args_cli.measure_last_s,
        },
        "criteria": {
            "drag_threshold_mps": args_cli.drag_threshold,
            "tilt_limit_deg": args_cli.tilt_limit,
            "height_limit_m": args_cli.height_limit,
        },
        "policies": {},
    }

    for name, checkpoint in policies:
        print(f"\n[traction] 扫 {name} ← {checkpoint}")
        policy = sweep.load_policy(checkpoint)
        per_env = sweep.run_one(policy)
        rows = aggregate(conditions, per_env, args_cli.repeats)
        saturation = saturation_forces(rows)
        report["policies"][name] = {
            "checkpoint": str(checkpoint),
            "conditions": rows,
            "saturation_n": saturation,
        }
        print(render_table(name, rows, saturation))

    sweep.close()

    out = Path(args_cli.output) if args_cli.output else spec.TRACTION_LOG_DIR / "drag_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[traction] 报告 → {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
