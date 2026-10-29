#!/usr/bin/env python3
"""跨引擎 sim2sim：把 IsaacLab 训出来的策略放进 MuJoCo 参考模型里评估。

**为什么要跨引擎**：用辨识参数训的策略、再用同一套辨识参数的 IsaacLab 评估，是循环论证。
这里的参考模型是 MuJoCo 的 `corrected` —— 它被真机数据独立验证过
（`walk_replay` 里短时预测误差是 baseline 的 0.56 / 0.66 / 0.68 倍），
而且换了物理引擎，训练时的建模误差不会在评估里自我抵消。

参考模型 = pace_bpx_pro 的 `FloorSim` + 以下设定（和 landing_compare / walk_replay 的 corrected 一致）：
  * 关节参数取辨识 yaml 的标称值（armature / 粘滞 / 零偏 0 / 延迟 2 步）
  * `set_hard_friction()` —— MuJoCo 默认的摩擦约束太软，慢速下几乎不产生摩擦力
  * 整机质量 15.1 kg（只改 torso，惯量同比例缩放）
  * 膝摩擦按负载更新 F = f0 + mu·L（默认 1.0 + 0.26·L）

**只读引用 pace_bpx_pro，不修改它的任何文件。** 本脚本不需要 Isaac Sim，可以和训练并行跑。

    conda activate bpx_rma
    cd ~/intern/ZGYD/bpx
    python scripts/sim2sim.py \
        --policy 辨识=logs/rsl_rl/bpx_pro_flat/*_student/exported/policy.onnx \
        --policy 厂家=logs/rsl_rl/bpx_pro_flat_baseline/*_student/exported/policy.onnx
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_rma import spec  # noqa: E402


# ---------------------------------------------------------------------------
# 只读引用 pace_bpx_pro
# ---------------------------------------------------------------------------
def import_pace(pace_dir: str | None):
    candidates = [Path(pace_dir)] if pace_dir else []
    candidates += [
        PROJECT_DIR.parent / "pace" / "pace_bpx_pro",
        Path.home() / "intern" / "ZGYD" / "pace" / "pace_bpx_pro",
    ]
    for c in candidates:
        if (c / "pace_bpx_pro" / "mujoco_floor.py").exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            from pace_bpx_pro import mujoco_floor  # noqa: PLC0415

            return mujoco_floor, c
    raise FileNotFoundError(
        "找不到 pace_bpx_pro（需要 pace_bpx_pro/mujoco_floor.py）。用 --pace-dir 指定。"
    )


# ---------------------------------------------------------------------------
# 指令序列：覆盖训练范围，并在高速段多给时间（可分辨性分析显示惯量/阻尼只在高速下显现）
# ---------------------------------------------------------------------------
PRESETS: dict[str, list[tuple[str, float, tuple[float, float, float]]]] = {
    # 覆盖训练范围的综合序列
    "mixed": [
        ("站立",        3.0, (0.0, 0.0, 0.0)),
        ("前进 0.8",    5.0, (0.8, 0.0, 0.0)),
        ("前进 1.5",    6.0, (1.5, 0.0, 0.0)),
        ("后退 0.8",    4.0, (-0.8, 0.0, 0.0)),
        ("侧移 0.5",    4.0, (0.0, 0.5, 0.0)),
        ("转向 1.0",    4.0, (0.0, 0.0, 1.0)),
        ("前进+转向",   4.0, (1.0, 0.0, 0.5)),
    ],
    # 直线快走：段给长、速度逐级加到训练上限。
    # 可分辨性分析显示惯量/阻尼只在高速下才显现（膝惯量占比 0.9% @0.5m/s -> 9.4% @1.5m/s），
    # 所以这组专门压高速段，而且每段跳过起步瞬态。
    "straight": [
        ("站立",        3.0, (0.0, 0.0, 0.0)),
        ("前进 0.5",   10.0, (0.5, 0.0, 0.0)),
        ("前进 0.8",   10.0, (0.8, 0.0, 0.0)),
        ("前进 1.2",   10.0, (1.2, 0.0, 0.0)),
        ("前进 1.5",   12.0, (1.5, 0.0, 0.0)),
    ],
    # 极速扫描：一路加到训练上限以上，看实际能跑到多少、哪里饱和。
    # 3.0 以上是训练分布之外（外推），策略可能只是追不上，也可能摔 —— 这正是要测的。
    "topspeed": [
        ("站立",       4.0, (0.0, 0.0, 0.0)),
        ("指令 1.5",   6.0, (1.5, 0.0, 0.0)),
        ("指令 2.0",   6.0, (2.0, 0.0, 0.0)),
        ("指令 2.5",   6.0, (2.5, 0.0, 0.0)),
        ("指令 3.0",   6.0, (3.0, 0.0, 0.0)),
        ("指令 3.5",   6.0, (3.5, 0.0, 0.0)),
        ("指令 4.0",   6.0, (4.0, 0.0, 0.0)),
        ("指令 5.0",   6.0, (5.0, 0.0, 0.0)),
    ],
    # 极速扫描（新电机包络用）：转速上限从 URDF 的 15/22 抬到 30 之后零力矩墙挪到
    # 6.6 m/s，所以 topspeed 的 5.0 已经不够用，这组加到 6.5。段间不复位，逐级加速。
    "topspeed6": [
        ("站立",       4.0, (0.0, 0.0, 0.0)),
        ("指令 2.0",   6.0, (2.0, 0.0, 0.0)),
        ("指令 3.0",   6.0, (3.0, 0.0, 0.0)),
        ("指令 3.5",   6.0, (3.5, 0.0, 0.0)),
        ("指令 4.0",   6.0, (4.0, 0.0, 0.0)),
        ("指令 4.5",   6.0, (4.5, 0.0, 0.0)),
        ("指令 5.0",   6.0, (5.0, 0.0, 0.0)),
        ("指令 5.5",   6.0, (5.5, 0.0, 0.0)),
        ("指令 6.0",   6.0, (6.0, 0.0, 0.0)),
        ("指令 6.5",   6.0, (6.5, 0.0, 0.0)),
    ],
    # 长驻留安全验证：在高指令上各停 25 s，看会不会在后段塌。
    # topspeed6 每段只有 6 s，"6.0 不摔"这个结论只来自一次短采样，不足以当安全依据。
    # 用途：定手柄限幅。段间不复位，逐级加速进来（冷启动另有 coldstart 预设）。
    "dwell": [
        ("站立",       3.0, (0.0, 0.0, 0.0)),
        ("热身 3.0",   6.0, (3.0, 0.0, 0.0)),
        ("热身 4.5",   6.0, (4.5, 0.0, 0.0)),
        ("驻留 5.0",  25.0, (5.0, 0.0, 0.0)),
        ("驻留 5.5",  25.0, (5.5, 0.0, 0.0)),
        ("驻留 6.0",  25.0, (6.0, 0.0, 0.0)),
    ],
    # 冷启动：从站立**直接**给大指令，复刻"手柄一把推到底"的场景。
    # topspeed/topspeed6 是逐级加速、段间不复位，策略进每一段时已经在跑了，测不出这个。
    # 站桩（高指令下策略选择不动）只在冷启动时暴露：奖励侧 exp 跟踪核在大误差处饱和，
    # 从"站着"到"跑起来"这一路没有梯度。
    "coldstart": [
        ("站立 A",     4.0, (0.0, 0.0, 0.0)),
        ("冷启 6.0",  12.0, (6.0, 0.0, 0.0)),
        ("站立 B",     3.0, (0.0, 0.0, 0.0)),
        ("冷启 4.0",  12.0, (4.0, 0.0, 0.0)),
        ("站立 C",     3.0, (0.0, 0.0, 0.0)),
        ("冷启 2.0",  12.0, (2.0, 0.0, 0.0)),
    ],
    # 冷启动阈值细扫：coldstart 测出 4.0 能起步、6.0 站桩，这组把分界夹出来。
    # 用途：① 定手柄指令的安全上限（或斜坡率要覆盖到哪）；② 给 sprint9 做验收基线。
    "coldstart_fine": [
        ("站立 A",     4.0, (0.0, 0.0, 0.0)),
        ("冷启 4.5",  12.0, (4.5, 0.0, 0.0)),
        ("站立 B",     3.0, (0.0, 0.0, 0.0)),
        ("冷启 5.0",  12.0, (5.0, 0.0, 0.0)),
        ("站立 C",     3.0, (0.0, 0.0, 0.0)),
        ("冷启 5.5",  12.0, (5.5, 0.0, 0.0)),
    ],
    # 高速组专用（训练上限 3.0 m/s）。1.5 m/s 下两组打平，这组看更高速能不能分开。
    "straight_fast": [
        ("站立",        3.0, (0.0, 0.0, 0.0)),
        ("前进 1.0",    8.0, (1.0, 0.0, 0.0)),
        ("前进 1.5",    8.0, (1.5, 0.0, 0.0)),
        ("前进 2.0",    8.0, (2.0, 0.0, 0.0)),
        ("前进 2.5",    8.0, (2.5, 0.0, 0.0)),
        ("前进 3.0",   10.0, (3.0, 0.0, 0.0)),
    ],
}
COMMANDS = PRESETS["mixed"]


#: 观测噪声幅度（均匀分布的半宽），取 IsaacLab 四足速度任务的标准值。
#: 顺序和 spec.OBS_TERMS 一致；last_action 和 command 是己方已知量，不加噪声。
OBS_NOISE = {
    "joint_pos": 0.01,          # rad，编码器
    "joint_vel": 1.5,           # rad/s，速度估计（通常是最脏的一项）
    "base_ang_vel": 0.2,        # rad/s，陀螺仪
    "projected_gravity": 0.05,  # 由加速度计推的倾角
    "last_action": 0.0,
    "velocity_command": 0.0,
}


def noise_vector(scale: float) -> np.ndarray:
    """按 spec.OBS_TERMS 的顺序展开成 45 维的噪声半宽。"""
    out = []
    for name, dim in spec.OBS_TERMS:
        out += [OBS_NOISE[name] * scale] * dim
    return np.array(out, dtype=np.float32)


def quat_to_proj_gravity(w: float, x: float, y: float, z: float) -> np.ndarray:
    """R(body->world)^T · [0,0,-1]，和板子 `projected_gravity_xyzw()` 同一个公式。"""
    return np.array(
        [2.0 * (w * y - x * z), -2.0 * (w * x + y * z), -(1.0 - 2.0 * (x * x + y * y))],
        dtype=np.float32,
    )


def quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class OnnxPolicy:
    """板子上那套推理流程的离线复刻：45 维帧 + 25 帧 term-major 历史 -> 12 维动作。"""

    def __init__(self, path: Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = {i.name: list(i.shape) for i in self.sess.get_inputs()}
        want = {
            spec.ONNX_INPUT_POLICY: [1, spec.OBS_FRAME_SIZE],
            spec.ONNX_INPUT_HISTORY: [1, spec.HISTORY_INPUT_SIZE],
        }
        if names != want:
            raise ValueError(f"{path} 的输入不符部署契约：{names} != {want}")
        self.q0 = np.array(spec.DEFAULT_JOINT_POS, dtype=np.float32)
        self.slices = spec.obs_term_slices()
        self.reset()

    def reset(self) -> None:
        self.last_action = np.zeros(spec.NUM_JOINTS, dtype=np.float32)
        self.history: np.ndarray | None = None

    def frame(self, q: np.ndarray, dq: np.ndarray, gyro: np.ndarray,
              grav: np.ndarray, cmd: np.ndarray) -> np.ndarray:
        return np.concatenate([q - self.q0, dq, gyro, grav, self.last_action, cmd]).astype(np.float32)

    def act(self, frame: np.ndarray) -> np.ndarray:
        if self.history is None:                      # 复位：第一帧回填满 25 格（和板子一致）
            self.history = np.repeat(frame.reshape(1, -1), spec.HISTORY_LENGTH, axis=0)
        else:
            self.history[:-1] = self.history[1:]
            self.history[-1] = frame
        flat = np.concatenate(
            [self.history[:, b:e].reshape(-1) for b, e in self.slices]
        ).astype(np.float32)
        out = self.sess.run(
            None,
            {spec.ONNX_INPUT_POLICY: frame.reshape(1, -1),
             spec.ONNX_INPUT_HISTORY: flat.reshape(1, -1)},
        )[0].ravel()
        raw = np.clip(out, -spec.ACTION_CLIP, spec.ACTION_CLIP).astype(np.float32)
        self.last_action = raw
        return self.q0 + spec.ACTION_SCALE * raw


def build_reference_sim(mj_floor, args):
    """参考"真实"模型：辨识标称值 + 硬摩擦约束 + 15.1 kg + 膝负载摩擦。"""
    params = mj_floor.load_final_yaml(args.ref_params, sim_dt=spec.SIM_DT)
    sim = mj_floor.FloorSim(params, kp=spec.KP_POLICY, kd=spec.KD_POLICY, dt=spec.SIM_DT)
    sim.set_hard_friction()

    # 整机质量（同 landing_compare.set_total_mass）
    m, tid = sim.model, sim.torso_id
    new = float(m.body_mass[tid] + (args.mass - sim.total_mass))
    if new < 1.0:
        raise SystemExit(f"--mass {args.mass} 太小")
    m.body_inertia[tid] *= new / m.body_mass[tid]
    m.body_mass[tid] = new
    sim.mj.mj_setConst(m, sim.data)
    sim.total_mass = float(np.sum(m.body_mass))

    # FloorSim 的力矩包络读的是 pace 那边 spec 的 URDF 值（膝 50 N·m）。
    # 这里按本项目的口径覆盖掉，让评估和训练用同一套电机能力：
    # 力矩上限 = 手册的 33 N·m，堵转 = ×2，可用力矩随速度线性衰减。
    # 不改 pace 的任何文件，只改这个 sim 实例的属性。
    eff = np.array([spec.EFFORT_LIMIT[spec.joint_type(n)] for n in spec.JOINT_NAMES], float)
    sim.effort = eff
    sim.sat = eff * spec.SATURATION_FACTOR_FALLBACK
    vlim_by_type = dict(spec.VELOCITY_LIMIT)
    if args.vlim:                                   # --vlim 22,15 = 髋 22 / 膝 15（覆盖 spec）
        hip, knee = (float(v) for v in args.vlim.split(","))
        vlim_by_type = {"hip_roll": hip, "hip_pitch": hip, "knee": knee}
        print(f"[sim2sim] 转速上限覆盖为 髋 {hip} / 膝 {knee} rad/s（spec 里是 "
              f"{spec.VELOCITY_LIMIT['hip_pitch']} / {spec.VELOCITY_LIMIT['knee']}）")
    sim.vlim = np.array([vlim_by_type[spec.joint_type(n)] for n in spec.JOINT_NAMES], float)
    sim.vel_at_lim = sim.vlim * (1.0 + sim.effort / sim.sat)

    f0, mu = (float(v) for v in args.knee_fric.split(","))
    sim.set_load_friction(np.array(spec.KNEE_INDICES), f0=f0, mu=mu)
    return sim, params


def settle(sim, q0: np.ndarray, seconds: float = 1.5) -> None:
    """从略高处放下，PD 保持 q0 让它站稳，再开始评估。"""
    sim.reset(q0, base_height=0.45)
    for _ in range(int(seconds / spec.SIM_DT)):
        sim.step(q0, kp=spec.KP_POLICY, kd=spec.KD_POLICY)


def run(sim, policy: OnnxPolicy, decimation: int, noise: np.ndarray | None = None,
        rng: np.random.Generator | None = None, commands=None, seg_warmup: float = 0.0) -> dict:
    q0 = np.array(spec.DEFAULT_JOINT_POS)
    settle(sim, q0)
    policy.reset()

    knee = np.array(spec.KNEE_INDICES)
    out: dict[str, list] = {"seg": [], "speed": [], "lin_err": [], "ang_err": [], "height": [],
                            "knee_tau": [], "dq_knee": [], "dq_hip": [], "tilt": [], "fell": []}
    fell_at = None
    t = 0.0
    n_warm = int(seg_warmup * spec.POLICY_HZ)
    for name, dur, (vx, vy, wz) in (commands or COMMANDS):
        cmd = np.array([vx, vy, wz], dtype=np.float32)
        acc = {k: [] for k in ("speed", "lin_err", "ang_err", "height", "knee_tau",
                               "dq_knee", "dq_hip", "tilt")}
        for k_step in range(int(dur * spec.POLICY_HZ)):
            qw, qx, qy, qz = sim.data.qpos[sim.free_qpos + 3: sim.free_qpos + 7]
            grav = quat_to_proj_gravity(qw, qx, qy, qz)
            frame = policy.frame(sim.q_enc(), sim.dq(), sim.base_gyro(), grav, cmd)
            if noise is not None:
                # 噪声加在帧上、只采样一次，之后这一帧原样进历史 ——
                # 和真机完全一致（板子的 history[-1] 就是送进去的那一帧）
                frame = frame + rng.uniform(-noise, noise).astype(np.float32)
            target = policy.act(frame)
            for _ in range(decimation):
                sim.step(target, kp=spec.KP_POLICY, kd=spec.KD_POLICY)
            t += 1.0 / spec.POLICY_HZ

            R = quat_to_rot(qw, qx, qy, qz)
            v_body = R.T @ sim.data.qvel[sim.free_dof: sim.free_dof + 3]
            tilt_now = float(np.degrees(np.arccos(np.clip(-grav[2], -1.0, 1.0))))
            if fell_at is None and (tilt_now > 60.0 or sim.base_pos()[2] < 0.15):
                fell_at = t
            if k_step < n_warm:      # 跳过指令切换后的加速瞬态，只统计稳态
                continue
            acc["speed"].append(float(v_body[0]))          # 实际前进速度（机身系）
            acc["lin_err"].append(float(np.linalg.norm(v_body[:2] - cmd[:2])))
            dqv = sim.dq()
            acc["dq_knee"].append(float(np.abs(dqv[knee]).max()))
            acc["dq_hip"].append(float(np.abs(np.delete(dqv, knee)).max()))
            acc["ang_err"].append(abs(float(sim.base_gyro()[2] - cmd[2])))
            acc["height"].append(float(sim.base_pos()[2]))
            acc["knee_tau"].append(float(np.abs(sim.tau_applied[knee]).mean()))
            acc["tilt"].append(tilt_now)
        out["seg"].append(name)
        # seg_warmup 比段长还长时这一段会是空的，不要让它把整个扫描打断
        for k in ("speed", "lin_err", "ang_err", "height", "knee_tau", "tilt"):
            out[k].append(float(np.mean(acc[k])) if acc[k] else float("nan"))
        for k in ("dq_knee", "dq_hip"):          # 关节速度看峰值，不是均值
            out[k].append(float(np.quantile(acc[k], 0.99)) if acc[k] else float("nan"))
        out["fell"].append(fell_at is not None)
    out["fell_at"] = fell_at
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="跨引擎 sim2sim 评估（MuJoCo 参考模型）")
    p.add_argument("--policy", action="append", required=True,
                   help="标签=路径（可含通配符），可重复。例：辨识=logs/.../policy.onnx")
    p.add_argument("--pace-dir", default=None, help="pace_bpx_pro 目录（默认自动找）")
    p.add_argument("--ref-params", default=str(PROJECT_DIR / "data" / "bpx_pro_training_params.yaml"),
                   help="参考模型的关节参数 yaml（默认用辨识结果——它是对真机最好的估计）")
    p.add_argument("--mass", type=float, default=15.1)
    p.add_argument("--knee-fric", default="1.0,0.26", help="膝负载摩擦 f0,mu")
    p.add_argument("--vlim", default=None,
                   help="覆盖关节转速上限 '髋,膝'（如 22,15 复刻 URDF 旧包络）；不给则用 spec")
    p.add_argument("--noise", type=float, nargs="*", default=[0.0],
                   help="观测噪声倍率，可给多个一次跑完（例：--noise 0 0.5 1.0 2.0）。"
                        "1.0 = IsaacLab 四足标准噪声：关节角 ±0.01rad、关节速度 ±1.5rad/s、"
                        "陀螺 ±0.2rad/s、投影重力 ±0.05")
    p.add_argument("--preset", choices=sorted(PRESETS), default="mixed",
                   help="mixed = 覆盖训练范围的综合序列；straight = 直线逐级加速到上限")
    p.add_argument("--seg-warmup", dest="seg_warmup", type=float, default=0.0,
                   help="每段开头跳过多少秒不统计（去掉指令切换后的加速瞬态）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(PROJECT_DIR / "logs" / "sim2sim"))
    args = p.parse_args()

    mj_floor, pace_dir = import_pace(args.pace_dir)
    print(f"[sim2sim] 只读引用 pace：{pace_dir}")

    entries = []
    for item in args.policy:
        label, _, pat = item.partition("=")
        if not pat:
            label, pat = Path(item).parent.parent.name, item
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"找不到策略：{pat}")
        entries.append((label, Path(hits[-1])))

    cmds = PRESETS[args.preset]
    print(f"[sim2sim] 指令预设 {args.preset}（{len(cmds)} 段，每段跳过前 {args.seg_warmup}s 瞬态）")
    decim = spec.DECIMATION
    results = {}
    labels: list[str] = []
    for label, path in entries:
        for nz in args.noise:
            tag = label if nz == 0 else f"{label}+噪声{nz:g}x"
            labels.append(tag)
            sim, params = build_reference_sim(mj_floor, args)
            print(f"\n[sim2sim] {tag}  <-  {path}")
            print(f"          参考模型 {params['name']}：整机 {sim.total_mass:.2f} kg，"
                  f"延迟 {params['delay_steps']} 步，膝摩擦 {args.knee_fric.replace(',', ' + ')}·L，硬约束")
            nvec = noise_vector(nz) if nz > 0 else None
            rng = np.random.default_rng(args.seed)   # 同一个种子，噪声序列可复现
            results[tag] = run(sim, OnnxPolicy(path), decim, noise=nvec, rng=rng,
                               commands=cmds, seg_warmup=args.seg_warmup)
            results[tag]["onnx"] = str(path)

    # --- 表格 ---
    segs = results[labels[0]]["seg"]
    print(f"\n{'='*86}")
    print("跨引擎 sim2sim：IsaacLab 训练 -> MuJoCo corrected 模型评估")
    print(f"{'='*86}")
    for metric, unit, fmt in [("speed", "m/s 实际", "{:.3f}"), ("lin_err", "m/s", "{:.4f}"),
                              ("ang_err", "rad/s", "{:.4f}"), ("height", "m", "{:.3f}"),
                              ("knee_tau", "N·m", "{:.2f}"), ("dq_knee", "rad/s 峰值", "{:.2f}"),
                              ("dq_hip", "rad/s 峰值", "{:.2f}")]:
        print(f"\n{metric} [{unit}]")
        print(f"  {'段':<12}" + "".join(f"{lb:>14}" for lb in labels))
        for i, s in enumerate(segs):
            row = "".join(f"{fmt.format(results[lb][metric][i]):>14}" for lb in labels)
            print(f"  {s:<12}{row}")
        avg = "".join(f"{fmt.format(np.mean(results[lb][metric])):>14}" for lb in labels)
        print(f"  {'各段平均':<11}{avg}")

    print("\n摔倒：")
    for lb in labels:
        fa = results[lb]["fell_at"]
        print(f"  {lb:<12}{'第 %.1f s 摔倒' % fa if fa else '全程没摔'}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"commands": [c[0] for c in cmds], "preset": args.preset, "results": results,
                    "ref": {"mass": args.mass, "knee_fric": args.knee_fric,
                            "params": str(args.ref_params)}},
                   ensure_ascii=False, indent=1)
    )
    print(f"\n[sim2sim] 已写 {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
