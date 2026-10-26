#!/usr/bin/env python3
"""闭环测试：估计器 → 导纳映射 → 现有策略，串起来真跑一遍。

    conda activate bpx_rma
    python scripts/play_traction.py --headless \
        --policy logs/rsl_rl/bpx_pro_sprint4/2026-10-21_11-02-37_student

这是**判这条链路到底行不行**的那个测试。前面所有离线指标答的都是"力估得准不准"
（4.4 N），而真正要回答的是"机器人跟不跟得上、松绳后停不停得住、没人拉会不会自己走"。
这两件事不是一回事 —— 4.4 N 的误差折算成速度指令是 0.32 m/s，在 1.5 m/s 的量程上
占 21%，人可能根本感觉不到，也可能一拉就乱走。没跑过就是不知道。

## 四类场景，每类对应一个具体的怀疑

| 场景 | 测什么 | 判据 |
|---|---|---|
| `follow` | 人匀速走，机器人跟不跟得上 | 速度在力方向的投影 ≥0.03 m/s 且正向占比 ≥60%；离人的距离不发散 |
| `idle` | 人站着不动、绳子松着 | 机器人几乎不动（≤0.15 m/s）—— 零力误报 P95 是 5 N，这条有实打实的风险 |
| `release` | 走着走着松手 | 1 秒内停住（最后 1 s 平均 ≤0.15 m/s） |
| `turn` | 人拐弯 | 跟随方向能不能转过来 |

**绳子是会松会紧的**（见 `env/leash.py`），不是恒力。恒力场景里绳子永远紧着，
"追上→绳松→力掉→停→人走远→绳紧"这个物理回路根本不存在，而那正是最可能出问题的地方。
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

parser = argparse.ArgumentParser(description="牵引闭环测试")
parser.add_argument("--policy", type=str,
                    default="logs/rsl_rl/bpx_pro_sprint4/2026-10-21_11-02-37_student")
parser.add_argument("--estimator", type=str, default=None,
                    help="TorchScript，默认 spec.DEFAULT_ESTIMATOR_DIR/estimator.ts")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--envs-per-scene", dest="envs_per_scene", type=int, default=32)
parser.add_argument("--duration-s", dest="duration_s", type=float, default=12.0)
parser.add_argument("--handler-speed", dest="handler_speed", type=float, default=0.8,
                    help="牵引者走多快 (m/s)。人正常走路 1.2~1.4")
parser.add_argument("--oracle-force", dest="oracle_force", action="store_true",
                    help="用真值力代替估计器 —— 分清'链路设计问题'和'估计器不够准'")
parser.add_argument("--output", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401
from bpx_traction import spec  # noqa: E402
from bpx_traction.env.frame import FrameAssembler  # noqa: E402
from bpx_traction.env.leash import Leash, LeashParams  # noqa: E402
from bpx_traction.env.tether import TetherApplier  # noqa: E402
from bpx_traction.env import traction_cfg  # noqa: E402
from bpx_traction.mapping import AdmittanceMapping, AdmittanceParams  # noqa: E402

#: 四个场景。每个 env 固定属于一个场景，所有场景并行跑。
SCENES = ("follow", "idle", "release", "turn")


def load_agent_cfg():
    cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    return handle_deprecated_rsl_rl_cfg(cfg, _pkg_version("rsl-rl-lib"))


def resolve_checkpoint(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_DIR / path).resolve()
    if path.is_file():
        return path
    return Path(get_checkpoint_path(str(path.parent), path.name, args_cli.checkpoint))


class ClosedLoop:
    def __init__(self):
        self.num_envs = len(SCENES) * args_cli.envs_per_scene
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        traction_cfg.prepare_for_traction(
            env_cfg, num_envs=self.num_envs,
            episode_length_s=max(60.0, 4 * args_cli.duration_s),
        )
        self.env = RslRlVecEnvWrapper(
            gym.make(args_cli.task, cfg=env_cfg), clip_actions=load_agent_cfg().clip_actions
        )
        self.base = self.env.unwrapped
        self.robot = self.base.scene["robot"]
        self.device = self.base.device
        self.dt = self.base.step_dt

        self.frames = FrameAssembler(self.base)
        self.tether = TetherApplier(self.robot, warn_unconfirmed=False)
        self.tether.reset_offsets()          # 闭环评估用固定标称挂点
        self.leash = Leash(self.num_envs, self.device, LeashParams())

        # 每个 env 属于哪个场景
        self.scene_id = torch.arange(self.num_envs, device=self.device) // args_cli.envs_per_scene

        # 估计器：每个 env 一个映射实例（映射是有状态的，不能共用）
        self.estimator = None
        if not args_cli.oracle_force:
            path = Path(args_cli.estimator) if args_cli.estimator else (
                spec.DEFAULT_ESTIMATOR_DIR / "estimator.ts")
            self.estimator = torch.jit.load(str(path), map_location=self.device).eval()
            print(f"[traction] 估计器 {path}")
        else:
            print("[traction] ⚠️ oracle 模式：用真值力，不走估计器")

        self.params = AdmittanceParams.from_json()
        self.mappings = [AdmittanceMapping(self.params) for _ in range(self.num_envs)]
        self.history = torch.zeros(
            self.num_envs, spec.HISTORY_LENGTH, spec.FRAME_SIZE, device=self.device
        )

    # ----------------------------------------------------------------------
    def handler_velocity(self, t: float) -> torch.Tensor:
        """各场景下牵引者怎么走。"""
        v = torch.zeros(self.num_envs, 3, device=self.device)
        speed = args_cli.handler_speed
        for idx, name in enumerate(SCENES):
            mask = self.scene_id == idx
            if name == "idle":
                continue                                   # 站着不动，绳子松的
            if name == "turn":
                angle = math.radians(60.0) * min(t / args_cli.duration_s, 1.0)
                v[mask, 0] = speed * math.cos(angle)
                v[mask, 1] = speed * math.sin(angle)
            else:                                           # follow / release
                v[mask, 0] = speed
        return v

    def run(self) -> dict:
        steps = int(round(args_cli.duration_s / self.dt))
        release_step = steps // 2                           # release 场景中途松手

        agent_cfg = load_agent_cfg()
        is_distill = agent_cfg.class_name == "DistillationRunner"
        runner_cls = DistillationRunner if is_distill else OnPolicyRunner
        runner = runner_cls(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        checkpoint = resolve_checkpoint(args_cli.policy)
        if is_distill:
            runner.load(str(checkpoint),
                        load_cfg={"student": True, "teacher": True, "iteration": False})
        else:
            runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=agent_cfg.device)
        print(f"[traction] 策略 {checkpoint}")

        self.tether.clear()
        self.base.reset()
        obs = self.env.get_observations()
        self.leash.reset(self.tether.attach_point_w())
        for m in self.mappings:
            m.reset()
        frame = self.frames.frame()
        self.history[:] = frame.unsqueeze(1).expand(-1, spec.HISTORY_LENGTH, -1)

        rec = {k: [] for k in ("tension", "distance", "taut", "speed", "proj",
                               "cmd_speed", "force_err", "terminated")}

        with torch.no_grad():
            for step in range(steps):
                t = step * self.dt
                if step == release_step:
                    self.leash.detach(torch.nonzero(self.scene_id == SCENES.index("release")).flatten())

                self.leash.set_velocity(self.handler_velocity(t))
                force_w = self.leash.step(self.tether.attach_point_w(), self.dt)
                true_force_b = self.tether.apply_world(force_w)

                # —— 感知 ——
                if self.estimator is not None:
                    values, confidence = self.estimator(self.history)
                    fb, fe = spec.output_slice("force_b")
                    est_force_b = values[:, fb:fe]
                    conf = confidence[:, spec.CONFIDENCE_TERMS.index("force")]
                else:
                    est_force_b = true_force_b
                    conf = torch.ones(self.num_envs, device=self.device)

                # —— 映射（逐 env，映射是有状态的）——
                est_cpu = est_force_b.cpu().numpy()
                conf_cpu = conf.cpu().numpy()
                cmd = np.stack([
                    self.mappings[i].step(est_cpu[i], float(conf_cpu[i]), self.dt)
                    for i in range(self.num_envs)
                ])
                command = torch.from_numpy(cmd).float().to(self.device)
                self.base.command_manager.get_command("base_velocity")[:] = command

                # —— 策略 ——
                action = policy(obs)
                obs, _, dones, _ = self.env.step(action)

                # —— 推进历史 ——
                frame = self.frames.frame()
                self.history = torch.roll(self.history, shifts=-1, dims=1)
                self.history[:, -1] = frame
                if bool(dones.any()):
                    idx = torch.nonzero(dones.bool()).flatten()
                    self.history[idx] = frame[idx].unsqueeze(1).expand(-1, spec.HISTORY_LENGTH, -1)

                # —— 记录 ——
                vel_b = self.robot.data.root_lin_vel_b[:, :2]
                f_b = true_force_b[:, :2]
                f_norm = torch.linalg.norm(f_b, dim=1)
                direction = f_b / f_norm.clamp_min(1e-6).unsqueeze(1)
                rec["tension"].append(self.leash.last_tension_n.cpu())
                rec["distance"].append(self.leash.last_distance_m.cpu())
                rec["taut"].append(self.leash.last_taut.cpu())
                rec["speed"].append(torch.linalg.norm(vel_b, dim=1).cpu())
                rec["proj"].append((vel_b * direction).sum(dim=1).cpu())
                rec["cmd_speed"].append(torch.linalg.norm(command[:, :2], dim=1).cpu())
                rec["force_err"].append(
                    torch.linalg.norm(est_force_b[:, :2] - f_b, dim=1).cpu())
                rec["terminated"].append(dones.bool().cpu())

        self.tether.clear()
        self.env.close()
        return {k: torch.stack(v).numpy() for k, v in rec.items()}, release_step


def judge(rec: dict, release_step: int, scene_id: np.ndarray) -> dict:
    """按场景出行为判据。"""
    steps = len(rec["speed"])
    settle = int(2.0 / spec.SAMPLE_PERIOD_S)      # 前 2 秒是起步瞬态，不计
    out = {}

    for idx, name in enumerate(SCENES):
        m = scene_id == idx
        row: dict = {"n_envs": int(m.sum()),
                     "terminated": float(rec["terminated"][:, m].any(axis=0).mean())}

        if name == "idle":
            # 没人拉的时候，机器人该站着。零力误报 P95 是 5 N，这条最危险。
            window = slice(settle, steps)
            speed = rec["speed"][window][:, m]
            row["mean_speed_mps"] = float(speed.mean())
            row["p95_speed_mps"] = float(np.percentile(speed, 95))
            row["drift_m"] = float(speed.mean() * (steps - settle) * spec.SAMPLE_PERIOD_S)
            row["cmd_nonzero_pct"] = float(
                (rec["cmd_speed"][window][:, m] > 1e-6).mean() * 100)
            row["pass"] = bool(row["mean_speed_mps"] <= 0.15)

        elif name == "release":
            tail = slice(steps - int(1.0 / spec.SAMPLE_PERIOD_S), steps)
            row["final_speed_mps"] = float(rec["speed"][tail][:, m].mean())
            post = rec["speed"][release_step:][:, m].mean(axis=1)
            below = np.nonzero(post <= 0.15)[0]
            row["stop_delay_s"] = float(below[0] * spec.SAMPLE_PERIOD_S) if len(below) else None
            row["pass"] = bool(row["final_speed_mps"] <= 0.15)

        else:                                        # follow / turn
            window = slice(settle, steps)
            taut = rec["taut"][window][:, m]
            proj = rec["proj"][window][:, m]
            live = taut
            row["taut_pct"] = float(taut.mean() * 100)
            row["proj_mps"] = float(proj[live].mean()) if live.any() else 0.0
            row["proj_positive_pct"] = float((proj[live] > 0).mean() * 100) if live.any() else 0.0
            row["distance_start_m"] = float(rec["distance"][settle][m].mean())
            row["distance_end_m"] = float(rec["distance"][-1][m].mean())
            row["distance_growth_m"] = row["distance_end_m"] - row["distance_start_m"]
            # 松紧振荡：绳子状态翻转的频率
            flips = np.diff(taut.astype(np.int8), axis=0) != 0
            row["taut_flips_per_s"] = float(
                flips.sum() / max(taut.shape[1], 1) / (taut.shape[0] * spec.SAMPLE_PERIOD_S))
            row["pass"] = bool(
                row["proj_mps"] >= 0.03 and row["proj_positive_pct"] >= 60.0
                and row["distance_growth_m"] <= 1.0
            )

        row["force_err_mean_n"] = float(rec["force_err"][settle:][:, m].mean())
        out[name] = row
    return out


def render(verdict: dict) -> str:
    lines = []
    for name, row in verdict.items():
        mark = "通过" if row.get("pass") else "未过"
        lines.append(f"\n[{mark}] {name}   {row['n_envs']} env   "
                     f"终止率 {row['terminated']:.0%}   力误差 {row['force_err_mean_n']:.2f} N")
        for k, v in row.items():
            if k in ("n_envs", "pass", "terminated", "force_err_mean_n"):
                continue
            lines.append(f"        {k:<22s} {v if v is None else round(v, 3)}")
    passed = sum(bool(r.get("pass")) for r in verdict.values())
    lines.append(f"\n{passed}/{len(verdict)} 个场景通过")
    return "\n".join(lines)


def main() -> None:
    loop = ClosedLoop()
    print(f"[traction] {loop.leash.describe()}")
    print(f"[traction] 场景 {SCENES}，每个 {args_cli.envs_per_scene} env，"
          f"{args_cli.duration_s:.0f} s，牵引者 {args_cli.handler_speed:.1f} m/s")
    scene_id = loop.scene_id.cpu().numpy()
    rec, release_step = loop.run()
    verdict = judge(rec, release_step, scene_id)
    print(render(verdict))

    out = Path(args_cli.output) if args_cli.output else (
        spec.TRACTION_LOG_DIR / ("closed_loop_oracle.json" if args_cli.oracle_force
                                 else "closed_loop.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "policy": str(resolve_checkpoint(args_cli.policy)),
        "oracle_force": args_cli.oracle_force,
        "handler_speed_mps": args_cli.handler_speed,
        "duration_s": args_cli.duration_s,
        "admittance": {k: v for k, v in vars(AdmittanceParams.from_json()).items()},
        "scenes": verdict,
    }, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[traction] 报告 → {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
