"""力场景：方向向量、网格完整性、时间剖面。

这些是纯数学，但错了以后在数据里看不出来 —— 方位角定义反了只会让估计器学到一个
镜像的映射，训练损失一样漂亮，上真机才发现往左拉它往右走。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_traction import spec  # noqa: E402
from bpx_traction.data import scenarios  # noqa: E402
from bpx_traction.data.scenarios import (  # noqa: E402
    CommandSpec,
    ForceSpec,
    TetherProfile,
)


# --------------------------------------------------------------------------
# 方向约定
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "azimuth, expected",
    [
        (0.0, (1.0, 0.0, 0.0)),      # 正前 = +x
        (90.0, (0.0, 1.0, 0.0)),     # 正左 = +y
        (180.0, (-1.0, 0.0, 0.0)),   # 正后
        (270.0, (0.0, -1.0, 0.0)),   # 正右
    ],
)
def test_azimuth_convention(azimuth, expected):
    v = ForceSpec(10.0, azimuth).vector_w()
    assert np.allclose(v, np.array(expected) * 10.0, atol=1e-9)


@pytest.mark.parametrize("elevation, sign", [(90.0, +1.0), (-90.0, -1.0)])
def test_elevation_convention(elevation, sign):
    """+90° 仰角 = 把机器人往上提。"""
    v = ForceSpec(10.0, 0.0, elevation).vector_w()
    assert v[2] == pytest.approx(sign * 10.0)
    assert np.allclose(v[:2], 0.0, atol=1e-9)


def test_magnitude_preserved():
    for az in (0.0, 37.0, 123.0, 300.0):
        for el in (-60.0, -15.0, 0.0, 30.0):
            f = ForceSpec(23.0, az, el)
            assert np.linalg.norm(f.vector_w()) == pytest.approx(23.0)


def test_horizontal_component():
    """映射只看水平分量（Fz 估但不用），所以这个函数不能算错。"""
    assert ForceSpec(10.0, 0.0, 0.0).horizontal_n() == pytest.approx(10.0)
    assert ForceSpec(10.0, 0.0, 90.0).horizontal_n() == pytest.approx(0.0, abs=1e-9)
    assert ForceSpec(10.0, 0.0, 60.0).horizontal_n() == pytest.approx(5.0)


# --------------------------------------------------------------------------
# 网格
# --------------------------------------------------------------------------
def test_zero_force_is_not_duplicated_across_directions():
    """0 N 的方向是无意义的，八个方位角不该产生八格一模一样的条件。"""
    grid = scenarios.force_grid([0.0, 10.0], scenarios.AZIMUTH_8)
    zeros = [f for f in grid if f.magnitude_n == 0.0]
    assert len(zeros) == 1
    assert len(grid) == 1 + len(scenarios.AZIMUTH_8)


def test_vertical_force_is_not_duplicated_across_azimuths():
    grid = scenarios.force_grid([10.0], scenarios.AZIMUTH_8, elevations_deg=(90.0,))
    assert len(grid) == 1, "竖直方向上方位角是多余的"


def test_grid_is_a_full_outer_product():
    forces = scenarios.force_grid([5.0, 10.0], (0.0, 90.0))
    commands = [CommandSpec(0.0), CommandSpec(1.0)]
    grid = scenarios.condition_grid(forces, commands)
    assert len(grid) == len(forces) * len(commands)
    assert len({c.label() for c in grid}) == len(grid), "标签不唯一，聚合时会串格"


def test_command_grid_clips_to_training_distribution():
    """侧向上限 0.5 比前向的 1.5 小，斜向指令必须被夹住而不是外推。"""
    cmds = scenarios.command_grid(speeds_mps=(1.5,), headings_deg=(90.0,))
    assert len(cmds) == 1
    assert cmds[0].vy == pytest.approx(0.5), "vy 没夹到训练分布上限"


def test_command_grid_dedups_zero_speed():
    """speed=0 时四个 heading 是同一格。"""
    cmds = scenarios.command_grid(speeds_mps=(0.0,), headings_deg=(0.0, 90.0, 180.0, 270.0))
    assert len(cmds) == 1


def test_sweep_conditions_cover_every_cell():
    conds = scenarios.sweep_conditions()
    mags = {c.force.magnitude_n for c in conds}
    azis = {c.force.azimuth_deg for c in conds if c.force.magnitude_n > 0}
    cmds = {c.command.label() for c in conds}
    # 非零力 × 方向 × 指令，加上零力 × 指令
    assert len(conds) == (len(mags) - 1) * len(azis) * len(cmds) + len(cmds)


def test_design_range_is_covered_by_default_sweep():
    """默认扫描必须覆盖设计力范围，否则量不出饱和点。"""
    mags = {c.force.magnitude_n for c in scenarios.sweep_conditions()}
    assert max(mags) >= spec.FORCE_DESIGN_RANGE_N[1]


def test_assign_to_envs_round_robin():
    conds = scenarios.sweep_conditions([0.0, 10.0], (0.0, 180.0), [CommandSpec(0.0)])
    idx = scenarios.assign_to_envs(conds, num_envs=2 * len(conds))
    assert len(idx) == 2 * len(conds)
    counts = np.bincount(idx, minlength=len(conds))
    assert (counts == 2).all(), "重复份额没有均匀铺开"


# --------------------------------------------------------------------------
# 时间剖面
# --------------------------------------------------------------------------
def test_profile_timeline():
    p = TetherProfile("hold", ForceSpec(20.0, 0.0), settle_s=1.0, ramp_s=0.5,
                      hold_s=3.0, tail_s=2.0)
    assert p.duration_s() == pytest.approx(6.5)
    assert p.num_steps() == int(round(6.5 / spec.SAMPLE_PERIOD_S))
    assert np.allclose(p.force_at(0.5), 0.0)                      # settle 段无力
    assert p.force_at(1.25)[0] == pytest.approx(10.0)             # ramp 中点 = 半力
    assert p.force_at(3.0)[0] == pytest.approx(20.0)              # hold 段满力


def test_ramp_zero_is_a_step():
    p = TetherProfile("hold", ForceSpec(20.0, 0.0), settle_s=1.0, ramp_s=0.0)
    assert p.force_at(0.99)[0] == pytest.approx(0.0)
    assert p.force_at(1.01)[0] == pytest.approx(20.0)


def test_release_drops_to_zero():
    p = TetherProfile("release", ForceSpec(20.0, 0.0))
    event = p.event_time_s()
    assert event is not None
    assert p.force_at(event - 0.1)[0] == pytest.approx(20.0)
    assert np.allclose(p.force_at(event + 0.1), 0.0)


def test_reverse_flips_the_force():
    p = TetherProfile("reverse", ForceSpec(20.0, 0.0))
    event = p.event_time_s()
    before = p.force_at(event - 0.1)
    after = p.force_at(event + 0.1)
    assert np.allclose(after, -before, atol=1e-9), "reverse 默认应当是反向同幅值"


def test_reverse_honours_explicit_target():
    target = ForceSpec(30.0, 90.0)
    p = TetherProfile("reverse", ForceSpec(20.0, 0.0), reverse_to=target)
    after = p.force_at(p.event_time_s() + 0.1)
    assert np.allclose(after, target.vector_w())


def test_sweep_rotates_a_full_turn():
    p = TetherProfile("sweep", ForceSpec(20.0, 0.0), settle_s=0.0, ramp_s=0.0,
                      hold_s=4.0, tail_s=0.0, sweep_turns=1.0)
    start = p.force_at(0.0)
    quarter = p.force_at(1.0)
    half = p.force_at(2.0)
    assert np.allclose(start, [20.0, 0.0, 0.0], atol=1e-9)
    assert np.allclose(quarter, [0.0, 20.0, 0.0], atol=1e-6), "转了 1/4 圈应当指向 +y"
    assert np.allclose(half, [-20.0, 0.0, 0.0], atol=1e-6)


def test_sweep_keeps_magnitude_constant():
    """sweep 只转方向，幅值不动。去掉 settle / ramp 段，否则斜坡上的中间值会混进来。"""
    p = TetherProfile("sweep", ForceSpec(25.0, 0.0), settle_s=0.0, ramp_s=0.0,
                      hold_s=4.0, tail_s=0.0)
    traj = p.trajectory()
    assert np.allclose(np.linalg.norm(traj, axis=1), 25.0, atol=1e-6)
    # 方向确实在转：首尾不同向
    assert not np.allclose(traj[0] / 25.0, traj[len(traj) // 2] / 25.0, atol=1e-3)


def test_trajectory_shape_and_dtype():
    p = TetherProfile("release", ForceSpec(15.0, 45.0))
    traj = p.trajectory()
    assert traj.shape == (p.num_steps(), 3)
    assert np.isfinite(traj).all()


def test_transient_profiles_cover_four_kinds():
    kinds = {p.kind for p in scenarios.transient_profiles(ForceSpec(20.0, 0.0))}
    assert kinds == {"ramp", "release", "reverse", "sweep"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
