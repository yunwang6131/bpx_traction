"""采集循环：跑策略 + 施加牵引力 + 攒 25 帧历史 + 打标签。

脚本入口在 `scripts/collect_traction.py`，这里只放不依赖 argparse 的那部分，
方便单独测和复用。

## 三条容易静默出错的地方

**历史的时间方向。** 最旧在前、最新在末尾（`spec.HISTORY_NEWEST_LAST`）。
接反了不会报错，只会让估计器学一个延迟半秒的映射。

**复位之后的历史。** 参考实现是 `history[done] = 0` —— 往数据里塞一批全零历史，
那是分布外的输入。这里照板子 `prepare_policy_state()` 的做法：**用复位后的第一帧回填
全部 25 格**，并且在攒够 25 帧真实数据之前不出样本。

**`push_robot` 必须关。** 它是瞬时速度扰动，产生的本体响应和外力很像，
但它**没有对应的力标签** —— 留着就等于往数据里掺"有响应、标签却是零力"的样本，
直接教估计器低估。`prepare_for_traction()` 默认关掉它。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .. import spec
from ..env.frame import FrameAssembler
from ..env.tether import TetherApplier
from . import scenarios


#: profile_kind 列的枚举顺序。诊断脚本按它解码，别重排。
PROFILE_KINDS = ("hold", "ramp", "release", "reverse", "sweep")


@dataclass
class CollectionConfig:
    """一轮采集的参数。"""

    #: 每轮跑多少个控制步。要覆盖最长的力剖面。
    steps_per_round: int = 400
    #: 每多少步出一个样本。相邻窗口高度重叠，出太密只是把相关样本堆大。
    stride: int = 5
    #: 跑几轮。每轮重新抽条件、重新随机挂点、重新复位。
    rounds: int = 4
    #: 挂点是否逐 env 随机（见 `tether.randomize_offsets` 的取舍说明）。
    randomize_attach: bool = True
    #: 力的档位。上限按扛力扫描实测的 30 N，另外带一档 OOD（45 N）让置信度头学会说
    #: "我不知道"。
    #:
    #: **低档位加得密**（2/3/5/8）是因为死区就落在这个带里。第一次采集用
    #: `(0, 5, 10, …)` 的均匀档位，结果 2–5 N 只占 2% —— 0 和 5 之间没有档，
    #: 那个区间只在 ramp 的瞬间扫过。而死区要不要设在 3 N 还是 8 N、估计器能不能
    #: 把"轻轻一碰"和"没有力"分开，全靠这一段的数据。
    #: Go2 的验收报告里"2–5 N 方向尾部"也正是被点名的弱区。
    magnitudes_n: tuple[float, ...] = (
        0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0
    )
    #: 四类瞬态的配比，按出现概率。稳态样本便宜、瞬态样本贵，但瞬态才是难点。
    profile_weights: dict[str, float] = field(
        default_factory=lambda: {"hold": 0.3, "ramp": 0.2, "release": 0.2,
                                 "reverse": 0.15, "sweep": 0.15}
    )
    seed: int = 0


@dataclass
class Rollout:
    """一轮采集的产出。"""

    history: np.ndarray        # (M, 25, 61) float32
    targets: np.ndarray        # (M, 15)     float32
    env_id: np.ndarray         # (M,)  int32  —— 按它切 train/val/test
    policy_id: np.ndarray      # (M,)  int32
    attach_offset: np.ndarray  # (M, 3) float32  挂点消融用
    profile_kind: np.ndarray   # (M,)  int8   力剖面类型（PROFILE_KINDS 的下标），诊断用
    since_change: np.ndarray   # (M,)  int32  距力上次变化（>0.5 N）多少步 —— 零力误报
                               #              要按"松手后多久"分层，没有它就只能猜

    def __len__(self) -> int:
        return len(self.history)


class TractionCollector:
    """把一个已经建好的 env 变成数据源。"""

    def __init__(self, env, cfg: CollectionConfig):
        self.env = env
        self.base = env.unwrapped
        self.cfg = cfg
        self.device = self.base.device
        self.num_envs = self.base.num_envs

        self.frames = FrameAssembler(self.base)
        self.tether = TetherApplier(self.base.scene["robot"])
        self.rng = np.random.default_rng(cfg.seed)

        #: (N, 25, 61) 环形缓冲。最旧在前、最新在末尾。
        self.history = torch.zeros(
            self.num_envs, spec.HISTORY_LENGTH, spec.FRAME_SIZE, device=self.device
        )
        #: 每个 env 自复位以来攒了多少帧真实数据。不够 25 不出样本。
        self.valid_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ----------------------------------------------------------------------
    def _sample_conditions(self) -> tuple[list[scenarios.TetherProfile], torch.Tensor]:
        """给每个 env 抽一个力剖面和一个速度指令。

        力和指令**各自独立抽，但都抽满网格**：力的档位、方向、剖面类型是正交的，
        指令覆盖策略的训练分布。这样 (力大小 × 力方向 × 指令大小 × 指令方向) 的格子
        随轮次累积会被填满，而不是像独立随机采样那样在四维里留下大片空白。
        """
        cfg = self.cfg
        kinds = list(cfg.profile_weights)
        probs = np.array([cfg.profile_weights[k] for k in kinds], dtype=float)
        probs /= probs.sum()

        profiles: list[scenarios.TetherProfile] = []
        for _ in range(self.num_envs):
            mag = float(self.rng.choice(cfg.magnitudes_n))
            az = float(self.rng.choice(scenarios.AZIMUTH_8))
            # 仰角：大部分水平（映射只用水平分量），留一成做上下拉扯
            el = float(self.rng.choice([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, -30.0]))
            kind = str(self.rng.choice(kinds, p=probs))
            profiles.append(
                scenarios.TetherProfile(
                    kind=kind,
                    peak=scenarios.ForceSpec(mag, az, el),
                    settle_s=float(self.rng.uniform(0.4, 1.0)),
                    ramp_s=float(self.rng.uniform(0.0, 0.8)),
                    hold_s=float(self.rng.uniform(1.5, 4.0)),
                    tail_s=float(self.rng.uniform(1.0, 2.5)),
                )
            )

        ranges = self.base.cfg.commands.base_velocity.ranges
        speed_max = float(max(abs(v) for v in ranges.lin_vel_x))
        commands = scenarios.command_grid(
            speeds_mps=(0.0, 0.25 * speed_max, 0.5 * speed_max, 0.75 * speed_max, speed_max),
            vx_limit=float(ranges.lin_vel_x[1]),
            vy_limit=float(ranges.lin_vel_y[1]),
            yaw_rates=(0.0, float(ranges.ang_vel_z[1]) * 0.5, float(ranges.ang_vel_z[0]) * 0.5),
        )
        picks = self.rng.integers(0, len(commands), size=self.num_envs)
        command = torch.tensor(
            np.stack([commands[i].vector() for i in picks]), dtype=torch.float32,
            device=self.device,
        )
        return profiles, command

    def _refill_history(self, frame: torch.Tensor, env_ids: torch.Tensor) -> None:
        """复位后用第一帧回填全部 25 格 —— 和板子 `prepare_policy_state()` 一致。

        不能填零：全零历史是分布外的输入，而且每次摔倒都会往数据里塞一批。
        """
        self.history[env_ids] = frame[env_ids].unsqueeze(1).expand(
            -1, spec.HISTORY_LENGTH, -1
        )
        self.valid_steps[env_ids] = 1

    def _push(self, frame: torch.Tensor) -> None:
        """推进一帧：整体左移，新帧放末尾。"""
        self.history = torch.roll(self.history, shifts=-1, dims=1)
        self.history[:, -1] = frame
        self.valid_steps = torch.clamp(self.valid_steps + 1, max=spec.HISTORY_LENGTH)

    # ----------------------------------------------------------------------
    def run_round(self, policy, policy_id: int = 0) -> Rollout:
        """跑一轮，返回这一轮采到的样本。"""
        cfg = self.cfg
        dt = self.base.step_dt

        if cfg.randomize_attach:
            self.tether.randomize_offsets()
        else:
            self.tether.reset_offsets()

        profiles, command = self._sample_conditions()
        force_table = torch.tensor(
            np.stack([
                np.stack([p.force_at(k * dt) for p in profiles])      # (N, 3)
                for k in range(cfg.steps_per_round)
            ]),
            dtype=torch.float32, device=self.device,
        )                                                            # (T, N, 3)

        kind_of_env = torch.tensor(
            [PROFILE_KINDS.index(p.kind) for p in profiles],
            dtype=torch.int8, device=self.device,
        )
        # 距力上次变化多少步：力差 >0.5 N 算一次变化。逐步累计，和样本一起落盘。
        force_delta = torch.linalg.norm(force_table[1:] - force_table[:-1], dim=2) > 0.5
        since_change = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.tether.clear()
        self.base.reset()
        obs = self.env.get_observations()
        self.valid_steps.zero_()

        hist_chunks, target_chunks, env_chunks = [], [], []
        kind_chunks, since_chunks = [], []
        offsets = self.tether.offsets_b.clone()

        with torch.no_grad():
            for step in range(cfg.steps_per_round):
                # 指令每步都写：command manager 会按 resampling_time_range 覆盖
                self.base.command_manager.get_command("base_velocity")[:] = command

                force_b = self.tether.apply_world(force_table[step])
                frame = self.frames.frame()

                fresh = self.valid_steps == 0
                if bool(fresh.any()):
                    self._refill_history(frame, torch.nonzero(fresh).flatten())
                else:
                    self._push(frame)

                if step > 0:
                    since_change = torch.where(
                        force_delta[step - 1], torch.zeros_like(since_change), since_change + 1
                    )

                ready = self.valid_steps >= spec.HISTORY_LENGTH
                if step % cfg.stride == 0 and bool(ready.any()):
                    idx = torch.nonzero(ready).flatten()
                    hist_chunks.append(self.history[idx].clone().cpu())
                    target_chunks.append(self.frames.labels(force_b)[idx].cpu())
                    env_chunks.append(idx.cpu())
                    kind_chunks.append(kind_of_env[idx].cpu())
                    since_chunks.append(since_change[idx].to(torch.int32).cpu())

                action = policy(obs)
                obs, _, dones, _ = self.env.step(action)
                if bool(dones.any()):
                    # 摔倒/超时的 env 历史作废，下一步用新的第一帧回填
                    self.valid_steps[dones.bool()] = 0

        self.tether.clear()

        if not hist_chunks:
            return Rollout(
                np.zeros((0, spec.HISTORY_LENGTH, spec.FRAME_SIZE), np.float32),
                np.zeros((0, spec.OUTPUT_SIZE), np.float32),
                np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 3), np.float32),
                np.zeros(0, np.int8), np.zeros(0, np.int32),
            )

        env_id = torch.cat(env_chunks).numpy().astype(np.int32)
        return Rollout(
            history=torch.cat(hist_chunks).numpy().astype(np.float32),
            targets=torch.cat(target_chunks).numpy().astype(np.float32),
            env_id=env_id,
            policy_id=np.full(len(env_id), policy_id, dtype=np.int32),
            attach_offset=offsets.cpu().numpy()[env_id].astype(np.float32),
            profile_kind=torch.cat(kind_chunks).numpy(),
            since_change=torch.cat(since_chunks).numpy(),
        )


# ---------------------------------------------------------------------------
def concat(rollouts: list[Rollout]) -> Rollout:
    if not rollouts:
        raise ValueError("没有可合并的 rollout")
    return Rollout(
        history=np.concatenate([r.history for r in rollouts]),
        targets=np.concatenate([r.targets for r in rollouts]),
        env_id=np.concatenate([r.env_id for r in rollouts]),
        policy_id=np.concatenate([r.policy_id for r in rollouts]),
        attach_offset=np.concatenate([r.attach_offset for r in rollouts]),
        profile_kind=np.concatenate([r.profile_kind for r in rollouts]),
        since_change=np.concatenate([r.since_change for r in rollouts]),
    )


def split_by_env(data: Rollout, num_envs: int, fractions=(0.70, 0.15)) -> dict[str, np.ndarray]:
    """**按 env 索引**切 train/val/test，不是按样本随机切。

    相邻窗口有 24/25 帧是重叠的，随机切会让几乎相同的样本同时落进训练集和测试集，
    测出来的泛化是假的。按 env 切，同一个 env 的全部样本留在同一份里。
    """
    train_cut = int(num_envs * fractions[0])
    val_cut = int(num_envs * (fractions[0] + fractions[1]))
    return {
        "train": data.env_id < train_cut,
        "validation": (data.env_id >= train_cut) & (data.env_id < val_cut),
        "test": data.env_id >= val_cut,
    }


def summarize(data: Rollout) -> str:
    force = data.targets[:, slice(*spec.output_slice("force_b"))]
    magnitude = np.linalg.norm(force, axis=1)
    bins = [0.0, 2.0, 5.0, 15.0, 30.0, 1e9]
    counts = [int(((magnitude >= lo) & (magnitude < hi)).sum())
              for lo, hi in zip(bins[:-1], bins[1:])]
    labels = ["<2N(零力)", "2–5N", "5–15N", "15–30N", ">30N"]
    total = max(len(data), 1)
    lines = [
        f"样本 {len(data):,}   env {len(np.unique(data.env_id))}   "
        f"策略 {len(np.unique(data.policy_id))}",
        "力的分布：" + "  ".join(
            f"{l} {c:,}({c / total:.0%})" for l, c in zip(labels, counts)
        ),
        f"挂点跨度：x[{data.attach_offset[:, 0].min():+.3f},{data.attach_offset[:, 0].max():+.3f}] "
        f"y[{data.attach_offset[:, 1].min():+.3f},{data.attach_offset[:, 1].max():+.3f}] "
        f"z[{data.attach_offset[:, 2].min():+.3f},{data.attach_offset[:, 2].max():+.3f}] m",
    ]
    return "\n".join(lines)
