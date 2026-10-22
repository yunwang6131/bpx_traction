"""牵引力场景：方向网格 + 时间剖面。

纯 numpy，不依赖 isaaclab —— 场景表能离线生成、离线核对，不用启 Isaac Sim。

## 方向约定

世界系，机器人启动时朝 +x。方位角 `azimuth` 绕 +z 逆时针，所以：

    0° = 正前方(+x)   90° = 正左方(+y)   180° = 正后方   270° = 正右方

仰角 `elevation` 0° = 水平，+90° = 正上方（把机器人往上提）。

## 为什么用正交网格而不是独立随机

Go2 那份数据集的采集是各维独立随机采的，结果是：速度指令通道的 std 只有
`[0.139, 0.116, 0.015]` m/s、关节位置 std 只有 0.04–0.06 rad —— 基本都是**原地被拉**，
行走段几乎没有覆盖（见 `reference/PROVENANCE.md` §2.5）。

四维（力大小 × 力方向 × 指令大小 × 指令方向）独立随机采样，格子会空，而空掉的恰好是
部署时最常出现的组合（"一边走一边被拉"）。显式建网格能保证每格都有样本。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from .. import spec

# 八个水平方向：四正 + 四斜
AZIMUTH_8 = tuple(float(a) for a in range(0, 360, 45))
#: 六轴：±x, ±y, ±z。诊断用，不是网格的一部分。
AXIS_6 = (
    (0.0, 0.0), (180.0, 0.0),      # ±x
    (90.0, 0.0), (270.0, 0.0),     # ±y
    (0.0, 90.0), (0.0, -90.0),     # ±z（仰角 ±90 时方位角无意义）
)


@dataclass(frozen=True)
class ForceSpec:
    """一个恒定的牵引力：幅值 + 方向。"""

    magnitude_n: float
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    def vector_w(self) -> np.ndarray:
        """世界系的 (3,) 力向量。"""
        az = math.radians(self.azimuth_deg)
        el = math.radians(self.elevation_deg)
        horizontal = math.cos(el)
        return self.magnitude_n * np.array(
            [horizontal * math.cos(az), horizontal * math.sin(az), math.sin(el)],
            dtype=np.float64,
        )

    def horizontal_n(self) -> float:
        """水平分量的大小 —— 映射只看这个（Fz 估但不用）。"""
        return abs(self.magnitude_n * math.cos(math.radians(self.elevation_deg)))

    def label(self) -> str:
        if abs(self.elevation_deg) >= 89.9:
            return f"{self.magnitude_n:.0f}N_{'up' if self.elevation_deg > 0 else 'down'}"
        if self.elevation_deg == 0.0:
            return f"{self.magnitude_n:.0f}N_az{self.azimuth_deg:.0f}"
        return f"{self.magnitude_n:.0f}N_az{self.azimuth_deg:.0f}_el{self.elevation_deg:+.0f}"


@dataclass(frozen=True)
class CommandSpec:
    """一个速度指令。牵引层自己发的是 (vx, vy, 0)，但采数据时要覆盖完整分布。"""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0

    def vector(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.wz], dtype=np.float64)

    def label(self) -> str:
        if self.vx == self.vy == self.wz == 0.0:
            return "stand"
        return f"v{self.vx:+.1f},{self.vy:+.1f},w{self.wz:+.1f}"


@dataclass(frozen=True)
class Condition:
    """网格里的一格：力 + 指令。扛力扫描里一个 env 跑一格。"""

    force: ForceSpec
    command: CommandSpec

    def label(self) -> str:
        return f"{self.force.label()}|{self.command.label()}"


# ---------------------------------------------------------------------------
# 网格
# ---------------------------------------------------------------------------
def force_grid(
    magnitudes_n: Sequence[float],
    azimuths_deg: Sequence[float] = AZIMUTH_8,
    elevations_deg: Sequence[float] = (0.0,),
) -> list[ForceSpec]:
    """力的方向网格。幅值 0 时只保留一个（方向无意义）。"""
    out: list[ForceSpec] = []
    for mag in magnitudes_n:
        if mag == 0.0:
            out.append(ForceSpec(0.0, 0.0, 0.0))
            continue
        for el in elevations_deg:
            if abs(el) >= 89.9:                       # 竖直方向上方位角是多余的
                out.append(ForceSpec(mag, 0.0, el))
                continue
            for az in azimuths_deg:
                out.append(ForceSpec(mag, az, el))
    return out


def command_grid(
    speeds_mps: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
    headings_deg: Sequence[float] = (0.0, 90.0, 180.0, 270.0),
    yaw_rates: Sequence[float] = (0.0,),
    vy_limit: float = 0.5,
    vx_limit: float = 1.5,
) -> list[CommandSpec]:
    """速度指令网格。按**策略的训练分布**夹，不是按板子限幅。

    默认的 vx/vy 上限对应 `BpxProFlatEnvCfg` 的 ``lin_vel_x=(-1.0, 1.5)`` /
    ``lin_vel_y=(-0.5, 0.5)``。换基底（比如 Sprint7 的 vx 到 5.0）时要跟着改，
    否则采到的数据覆盖不了策略实际会走的速度段。
    """
    out: list[CommandSpec] = []
    seen: set[tuple[float, float, float]] = set()
    for speed in speeds_mps:
        for heading in headings_deg:
            vx = float(np.clip(speed * math.cos(math.radians(heading)), -vx_limit, vx_limit))
            vy = float(np.clip(speed * math.sin(math.radians(heading)), -vy_limit, vy_limit))
            for wz in yaw_rates:
                key = (round(vx, 6), round(vy, 6), round(float(wz), 6))
                if key in seen:                        # speed=0 时四个 heading 是同一格
                    continue
                seen.add(key)
                out.append(CommandSpec(vx, vy, float(wz)))
    return out


def condition_grid(
    forces: Sequence[ForceSpec],
    commands: Sequence[CommandSpec],
) -> list[Condition]:
    """力 × 指令的完整外积。这就是"正交网格"—— 每一格都有样本，不留空。"""
    return [Condition(f, c) for f in forces for c in commands]


def sweep_conditions(
    magnitudes_n: Sequence[float] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0),
    azimuths_deg: Sequence[float] = AZIMUTH_8,
    commands: Sequence[CommandSpec] | None = None,
) -> list[Condition]:
    """扛力扫描用的网格：水平力 × 8 方向 × 几个代表性指令。

    默认指令取"站立 / 慢走 / 常速前进" —— 扛力能力在站着和走着时可能很不一样，
    站着是静力平衡问题，走着还要同时维持步态。
    """
    if commands is None:
        commands = [CommandSpec(0.0, 0.0, 0.0), CommandSpec(0.5, 0.0, 0.0), CommandSpec(1.0, 0.0, 0.0)]
    return condition_grid(force_grid(magnitudes_n, azimuths_deg), commands)


# ---------------------------------------------------------------------------
# 时间剖面
# ---------------------------------------------------------------------------
@dataclass
class TetherProfile:
    """力随时间怎么变。四类瞬态照搬 Go2 的验收场景（见 PROVENANCE §3）。

    所有时间单位是秒，内部按 `spec.SAMPLE_PERIOD_S` 离散。
    """

    kind: str                       # hold / ramp / release / reverse / sweep
    peak: ForceSpec
    settle_s: float = 1.0           # 开场无力，让步态稳下来
    ramp_s: float = 0.5             # 加载用多久（0 = 阶跃）
    hold_s: float = 3.0             # 保持多久
    tail_s: float = 2.0             # 事件之后再跑多久（看恢复）
    reverse_to: ForceSpec | None = None   # kind=reverse 时切到哪个力
    sweep_turns: float = 1.0        # kind=sweep 时方位角转几圈

    def duration_s(self) -> float:
        return self.settle_s + self.ramp_s + self.hold_s + self.tail_s

    def num_steps(self) -> int:
        return int(round(self.duration_s() / spec.SAMPLE_PERIOD_S))

    def event_time_s(self) -> float | None:
        """release / reverse 的事件时刻 —— 瞬态指标都以它为原点。"""
        if self.kind in ("release", "reverse"):
            return self.settle_s + self.ramp_s + self.hold_s
        return None

    def force_at(self, t: float) -> np.ndarray:
        """`t` 秒时的世界系力 (3,)。"""
        peak = self.peak.vector_w()
        load_start, load_end = self.settle_s, self.settle_s + self.ramp_s
        hold_end = load_end + self.hold_s

        if t < load_start:
            return np.zeros(3)
        if t < load_end:
            frac = 1.0 if self.ramp_s <= 0 else (t - load_start) / self.ramp_s
            return peak * frac
        if t < hold_end or self.kind in ("hold", "ramp"):
            if self.kind == "sweep":
                return self._sweep_force(t, load_end)
            return peak
        # hold 之后
        if self.kind == "release":
            return np.zeros(3)
        if self.kind == "reverse":
            other = self.reverse_to or ForceSpec(
                self.peak.magnitude_n,
                (self.peak.azimuth_deg + 180.0) % 360.0,
                -self.peak.elevation_deg,
            )
            return other.vector_w()
        if self.kind == "sweep":
            return self._sweep_force(t, load_end)
        return peak

    def _sweep_force(self, t: float, start_s: float) -> np.ndarray:
        """方位角连续旋转 —— 对应 Go2 的 Full3D 场景，最难的一类。"""
        span = max(self.hold_s, 1e-6)
        frac = (t - start_s) / span
        azimuth = self.peak.azimuth_deg + 360.0 * self.sweep_turns * frac
        return ForceSpec(self.peak.magnitude_n, azimuth, self.peak.elevation_deg).vector_w()

    def trajectory(self) -> np.ndarray:
        """整条剖面，形状 (num_steps, 3)，世界系。"""
        dt = spec.SAMPLE_PERIOD_S
        return np.stack([self.force_at(k * dt) for k in range(self.num_steps())])


def transient_profiles(peak: ForceSpec, **kwargs) -> list[TetherProfile]:
    """一个力对应的四类瞬态，采数据时每格都要覆盖。"""
    return [
        TetherProfile("ramp", peak, **kwargs),
        TetherProfile("release", peak, **kwargs),
        TetherProfile("reverse", peak, **kwargs),
        TetherProfile("sweep", peak, **kwargs),
    ]


# ---------------------------------------------------------------------------
# 分配到并行环境
# ---------------------------------------------------------------------------
def assign_to_envs(conditions: Sequence[Condition], num_envs: int) -> np.ndarray:
    """把网格铺到 `num_envs` 个并行环境上，返回每个 env 的条件下标 (num_envs,)。

    条件数多于 env 数时循环分配（要跑多轮才能覆盖全）；少于时重复，
    重复份额在统计时按条件分组平均，不会把某一格算重。
    """
    if len(conditions) == 0:
        raise ValueError("条件网格是空的")
    return np.arange(num_envs, dtype=np.int64) % len(conditions)


def describe(conditions: Sequence[Condition]) -> str:
    mags = sorted({c.force.magnitude_n for c in conditions})
    azis = sorted({c.force.azimuth_deg for c in conditions if c.force.magnitude_n > 0})
    cmds = sorted({c.command.label() for c in conditions})
    return (
        f"{len(conditions)} 格 = "
        f"力 {len(mags)} 档 {mags} N × 方向 {len(azis)} 个 × 指令 {len(cmds)} 种 {cmds}"
    )


if __name__ == "__main__":
    conds = sweep_conditions()
    print(describe(conds))
    print()
    print("前几格：")
    for c in conds[:5]:
        print(f"  {c.label():<28s} 世界系力 {np.round(c.force.vector_w(), 2)}")
    print()
    p = TetherProfile("release", ForceSpec(20.0, 0.0))
    traj = p.trajectory()
    print(f"release 剖面：{p.duration_s():.1f} s = {p.num_steps()} 步，"
          f"事件在 {p.event_time_s():.1f} s")
    for t in (0.5, 1.2, 2.0, 4.6, 5.5):
        print(f"  t={t:4.1f}s  F={np.round(p.force_at(t), 1)}")
