#!/usr/bin/env python3
"""回放训好的策略，看它到底在怎么走。

    conda activate bpx_rma
    python scripts/play.py                       # 最新的 student（RMA）
    python scripts/play.py --agent rsl_rl_cfg_entry_point --load-run ".*_teacher"   # 看 teacher
    python scripts/play.py --headless --steps 500 --report                          # 不开窗口，只打指标

`--report` 会统计跟踪误差、机身高度、膝力矩、腾空时间 —— 和真机走路记录对比时用这几个。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="回放 BPX-Pro 策略")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_distillation_cfg_entry_point")
parser.add_argument("--load-run", dest="load_run", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default="model_.*.pt")
parser.add_argument("--experiment-name", dest="experiment_name", type=str, default=None,
                    help="覆盖实验名，用来回放对照组（如 bpx_pro_flat_baseline）")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--vx", type=float, default=0.5)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--side", action="store_true", help="正侧方视角 + 拉近，看步态细节用")
parser.add_argument("--report", action="store_true", help="最后打一份指标汇总")
parser.add_argument(
    "--random-cmd", dest="random_cmd", action="store_true",
    help="用训练时的完整指令分布（前后/侧移/转向随机组合），每 --cmd-period 秒换一次；忽略 --vx/--vy/--wz",
)
parser.add_argument("--cmd-period", dest="cmd_period", type=float, default=2.5, help="随机指令的重采周期 s")
parser.add_argument(
    "--ramp", type=float, default=0.0,
    help="vx 指令渐升速率 m/s²（0=恒定）。从 1.0 m/s 爬到 --vx；高速冷启动是 OOD，真机也走档位斜坡",
)
parser.add_argument("--video", action="store_true", help="录一段 mp4（会自动开 --headless --enable_cameras）")
parser.add_argument("--video-length", dest="video_length", type=int, default=500, help="录多少个控制步（50 Hz）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    # 离屏渲染，不开窗口，这样和正在跑的训练抢得少
    args_cli.headless = True
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from importlib.metadata import version as _pkg_version  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import bpx_rma  # noqa: E402, F401
from bpx_rma import spec  # noqa: E402


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.random_cmd:
        # 保留训练时的指令区间，只把重采周期缩短，好在一段短视频里看到多种指令
        env_cfg.commands.base_velocity.resampling_time_range = (args_cli.cmd_period, args_cli.cmd_period)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    else:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)

    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _pkg_version("rsl-rl-lib"))
    is_distill = agent_cfg.class_name == "DistillationRunner"
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    load_run = args_cli.load_run or (".*_student" if is_distill else ".*_teacher")

    log_root = (PROJECT_DIR / "logs" / "rsl_rl" / agent_cfg.experiment_name).resolve()
    ckpt = get_checkpoint_path(str(log_root), load_run, args_cli.checkpoint)
    print(f"[bpx] checkpoint：{ckpt}")

    if args_cli.video:
        # 相机跟着机器人走。固定机位在高速下没用 —— 3 m/s 十秒就是 30 米，早跑出画面了。
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        if args_cli.side:
            env_cfg.viewer.eye = (0.0, 2.0, 0.45)   # 正侧方、拉近、略高于机身，看腿部步态
            env_cfg.viewer.lookat = (0.0, 0.0, 0.25)
        else:
            env_cfg.viewer.eye = (2.2, 2.2, 1.0)      # origin_type 非 world 时，eye/lookat 是相对量
            env_cfg.viewer.lookat = (0.0, 0.0, 0.2)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_dir = str(PROJECT_DIR / "logs" / "videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda s: s == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            # 实验名必须进文件名，否则对照组会把辨识组的同名视频覆盖掉
            name_prefix=(
                f"{agent_cfg.experiment_name}_{'student' if is_distill else 'teacher'}"
                + ("_randcmd" if args_cli.random_cmd else f"_vx{args_cli.vx}")
            ),
        )
        print(f"[bpx] 录像输出目录：{video_dir}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = DistillationRunner if is_distill else OnPolicyRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if is_distill:
        runner.load(ckpt, load_cfg={"student": True, "teacher": True, "iteration": False})
    else:
        runner.load(ckpt)
    policy = runner.get_inference_policy(device=agent_cfg.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    knee_ids, _ = robot.find_joints(spec.KNEE_JOINT_NAMES, preserve_order=True)
    foot_ids, _ = robot.find_bodies(spec.FOOT_BODIES, preserve_order=True)
    stats: dict[str, list[torch.Tensor]] = {k: [] for k in ("lin_err", "ang_err", "height", "knee_tau")}
    foot_z: list[torch.Tensor] = []       # (N, 4) toe_link 世界 z，触地时 ≈ 0.0285（足端圆柱半径）
    all_ids, _ = robot.find_joints(spec.JOINT_NAMES, preserve_order=True)   # SDK 顺序 12 关节
    tau_all: list[torch.Tensor] = []      # (N, 12) 执行力矩，SDK 顺序
    grav: list[torch.Tensor] = []         # (N, 3) 机身系重力投影，判机身倾斜

    obs = env.get_observations()          # rsl_rl 5.x 的 wrapper 只返回 TensorDict
    step_dt = base.step_dt
    with torch.inference_mode():
        for i in range(args_cli.steps):
            if args_cli.ramp > 0:
                # 每步直接改指令张量（重采样会覆盖，所以每步都写）
                vx_now = min(1.0 + args_cli.ramp * i * step_dt, args_cli.vx)
                base.command_manager.get_command("base_velocity")[:, 0] = vx_now
                obs["policy"][:, 42] = vx_now          # 观测里的 cmd 项同步
                obs["history"][:, 42 * spec.HISTORY_LENGTH + spec.HISTORY_LENGTH - 1] = vx_now
            obs, _, _, _ = env.step(policy(obs))
            if args_cli.report:
                cmd = base.command_manager.get_command("base_velocity")
                stats["lin_err"].append((robot.data.root_lin_vel_b[:, :2] - cmd[:, :2]).norm(dim=-1))
                stats["ang_err"].append((robot.data.root_ang_vel_b[:, 2] - cmd[:, 2]).abs())
                stats.setdefault("vx_actual", []).append(robot.data.root_lin_vel_b[:, 0])  # 实际前向速度
                stats["height"].append(robot.data.root_pos_w[:, 2])
                stats["knee_tau"].append(robot.data.applied_torque[:, knee_ids].abs().mean(dim=-1))
                foot_z.append(robot.data.body_pos_w[:, foot_ids, 2].clone())
                tau_all.append(robot.data.applied_torque[:, all_ids].clone())
                grav.append(robot.data.projected_gravity_b.clone())

    if args_cli.report:
        label = (
            f"随机指令，每 {args_cli.cmd_period}s 换一次"
            if args_cli.random_cmd
            else f"指令 vx={args_cli.vx} vy={args_cli.vy} wz={args_cli.wz}"
        )
        print(f"\n--- {args_cli.steps} 步汇总（{label}）---")
        for k, v in stats.items():
            x = torch.stack(v)
            print(f"  {k:9s} 均值 {x.mean():7.4f}   标准差 {x.std():7.4f}")
        z = torch.stack(foot_z)                              # (T, N, 4)
        clearance = z - 0.0285                               # 足端圆柱底离地高度
        apex95 = torch.quantile(clearance, 0.95, dim=0)
        airborne = (clearance > 0.02).float().mean()
        # 逐摆动真峰值：对每段连续离地区间取 max，再对所有摆动求均值
        up = clearance > 0.015                               # (T, N, 4) 离地中
        c = clearance.permute(1, 2, 0).reshape(-1, clearance.shape[0])   # (N*4, T)
        m = up.permute(1, 2, 0).reshape(-1, up.shape[0])
        apexes = []
        for i in range(c.shape[0]):
            zi, mi = c[i], m[i]
            starts = (torch.nonzero(mi[1:] & ~mi[:-1]).flatten() + 1).tolist()
            ends = (torch.nonzero(~mi[1:] & mi[:-1]).flatten() + 1).tolist()
            if ends and starts and ends[0] <= starts[0]:
                ends = ends[1:]                              # 开场就在腾空，丢掉没头的半段
            for s, e in zip(starts, ends):
                if e > s + 2:                                # 丢掉 <3 步的抖动段
                    apexes.append(float(zi[s:e].max()))
        apex_true = sum(apexes) / max(len(apexes), 1)
        print(f"  foot_apex P95 {apex95.mean()*100:5.1f} cm   逐摆动真峰值均值 {apex_true*100:5.1f} cm"
              f"（{len(apexes)} 次摆动）  离地>2cm 时间占比 {airborne*100:4.1f}%")

        # 力矩饱和：训练仿真硬限 33，板子 clip 30 —— 踩在 30~33 的力矩上板会被砍
        tau = torch.stack(tau_all)                           # (T, N, 12) SDK 顺序
        amag = tau.abs()
        limit, board = spec.EFFORT_LIMIT["knee"], 30.0       # 三组限值相同
        groups = {"hip_roll": [0, 3, 6, 9], "hip_pitch": [1, 4, 7, 10], "knee": [2, 5, 8, 11]}
        print(f"  力矩 |τ|：  P99 {torch.quantile(amag, 0.99):.1f}   max {amag.max():.1f} N·m"
              f"   踩板子clip(≥30) {100*(amag >= board).float().mean():.1f}%"
              f"   逼近仿真限(≥32) {100*(amag >= 32).float().mean():.2f}%")
        for g, idx in groups.items():
            gm = amag[:, :, idx]
            print(f"      {g:9s} P99 {torch.quantile(gm, 0.99):4.1f}  ≥30 {100*(gm>=board).float().mean():4.1f}%")
        # 机身倾斜：正立时 proj_grav=(0,0,-1)；x=俯仰、y=横滚（左右高低）系统偏置
        g = torch.stack(grav)                                # (T, N, 3)
        gx, gy = g[..., 0].mean().item(), g[..., 1].mean().item()
        roll = torch.atan2(g[..., 1], -g[..., 2]).mean().item()
        print(f"  机身倾斜：proj_grav 均值 x(前后){gx:+.4f}  y(左右){gy:+.4f}   "
              f"等效横滚 {roll*57.3:+.2f}°（+=右低左高）")

        # ---- 步态分析：foot_z<0.045 视为触地（FOOT_RADIUS 0.0285 + 余量）----
        fz = torch.stack(foot_z)                             # (T, N, 4) = fl,fr,hl,hr toe
        contact = (fz < 0.045).float()                       # (T, N, 4)
        duty = contact.mean(dim=(0, 1))                      # 每脚占空比（支撑相时间比）
        dt = step_dt
        # 步频：每脚触地上升沿计数 / 时长
        rises = ((contact[1:] > 0.5) & (contact[:-1] < 0.5)).sum(dim=0).float()  # (N,4)
        stride_hz = (rises / (fz.shape[0] * dt)).mean(dim=0)  # 每脚平均步频
        # 对角 trot 同步：FL(0)&HR(3) 同相、FR(1)&HL(2) 同相，两对反相
        def corr(a, b):
            a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
            return ((a * b).mean(0) / (a.std(0) * b.std(0) + 1e-6)).mean().item()
        diag1 = corr(contact[:, :, 0], contact[:, :, 3])     # FL-HR
        diag2 = corr(contact[:, :, 1], contact[:, :, 2])     # FR-HL
        anti = corr(contact[:, :, 0], contact[:, :, 1])      # FL-FR（应反相=负）
        names = ["FL", "FR", "HL", "HR"]
        print(f"  步态：步频 {stride_hz.mean():.2f} Hz   占空比 " +
              " ".join(f"{n}{duty[i]*100:.0f}%" for i, n in enumerate(names)))
        print(f"        对角同步 FL-HR {diag1:+.2f} / FR-HL {diag2:+.2f}（+1=完美trot）  "
              f"同侧前腿 FL-FR {anti:+.2f}（-1=交替）")
        lr = (duty[[0, 2]].mean() - duty[[1, 3]].mean()).item()  # 左(FL,HL)-右(FR,HR) 占空比差
        print(f"        左右占空比差 {lr*100:+.1f}%（0=对称）")
        print("  （和真机走路记录对比时看这几项：线速度/角速度跟踪误差、机身高度、膝力矩）")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
