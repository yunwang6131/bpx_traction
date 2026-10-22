"""导纳映射的单测：给一条力的时间序列，断言指令序列。

这是整条链路里唯一不需要 GPU、不需要 Isaac Sim、不需要数据就能验的部分，
所以要验得狠一点。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_traction import spec                                    # noqa: E402
from bpx_traction.mapping import AdmittanceMapping, AdmittanceParams  # noqa: E402

DT = spec.SAMPLE_PERIOD_S


def run(mapping: AdmittanceMapping, force_xy, steps: int, conf: float = 1.0):
    """跑 `steps` 步恒力，返回 (steps, 3) 的指令序列。"""
    force = np.array([force_xy[0], force_xy[1], 0.0], dtype=float)
    return np.stack([mapping.step(force, conf) for _ in range(steps)])


# --------------------------------------------------------------------------
# 死区与迟滞
# --------------------------------------------------------------------------
def test_below_deadzone_stays_still():
    m = AdmittanceMapping()
    cmd = run(m, (m.params.deadzone_on_n - 0.5, 0.0), steps=200)
    assert np.allclose(cmd, 0.0), "死区以下必须完全不动"


def test_schmitt_hysteresis():
    """超过 on 才起步；起步后掉到 on 和 off 之间仍然走；掉到 off 以下才停。"""
    p = AdmittanceParams()
    m = AdmittanceMapping(p)

    run(m, (p.deadzone_on_n + 2.0, 0.0), steps=100)
    assert m.state.engaged, "超过 deadzone_on 应该起步"

    between = 0.5 * (p.deadzone_on_n + p.deadzone_off_n)
    run(m, (between, 0.0), steps=50)
    assert m.state.engaged, "迟滞带内应该保持走，否则边界会抖"

    run(m, (p.deadzone_off_n - 0.5, 0.0), steps=50)
    assert not m.state.engaged, "掉到 deadzone_off 以下应该停"


def test_hysteresis_prevents_chatter():
    """力在 deadzone_on 上下抖动时，指令不应该跟着来回开关。"""
    p = AdmittanceParams()
    m = AdmittanceMapping(p)
    run(m, (p.deadzone_on_n + 1.0, 0.0), steps=100)      # 先起步

    toggles = 0
    prev = m.state.engaged
    for k in range(200):                                  # 在 on 阈值上下 ±0.5 N 抖
        f = p.deadzone_on_n + (0.5 if k % 2 == 0 else -0.5)
        m.step(np.array([f, 0.0, 0.0]), 1.0)
        toggles += int(m.state.engaged != prev)
        prev = m.state.engaged
    assert toggles == 0, f"迟滞没挡住抖动，开关了 {toggles} 次"


# --------------------------------------------------------------------------
# 导纳本体
# --------------------------------------------------------------------------
def test_steady_state_matches_admittance():
    """恒力跑够久，速度应收敛到 steady_speed()。"""
    p = AdmittanceParams()
    m = AdmittanceMapping(p)
    force = 20.0
    cmd = run(m, (force, 0.0), steps=int(20 * p.tau_s() / DT))
    expected = p.steady_speed(force)
    assert cmd[-1, 0] == pytest.approx(expected, abs=1e-3), (
        f"稳态应为 F/B 型的 {expected:.4f}，实际 {cmd[-1, 0]:.4f}"
    )


def test_step_response_matches_zoh_solution():
    """阶跃响应必须逐点等于零阶保持的解析解 ``v[n] = (1 − α^n) · v_∞``。

    比"在 τ 处到 63%"强：那只在连续时间下成立，而 τ=127 ms 不是 20 ms 的整数倍，
    用它做断言会把离散化的舍入误差当成 bug。这里直接验离散解本身。
    用大 max_accel 跑，避免速率限幅污染。
    """
    p = AdmittanceParams(max_accel_mps2=1e6)
    m = AdmittanceMapping(p)
    force = 20.0
    cmd = run(m, (force, 0.0), steps=40)

    v_inf = p.steady_speed(force)
    alpha = math.exp(-DT / p.tau_s())
    expected = [(1.0 - alpha ** (n + 1)) * v_inf for n in range(len(cmd))]
    assert np.allclose(cmd[:, 0], expected, atol=1e-12), "离散响应偏离零阶保持解"

    # 顺带确认它确实是个时间常数为 τ 的一阶系统：t≈τ 处落在 63% 附近
    n_tau = int(round(p.tau_s() / DT))
    assert 0.55 < cmd[n_tau - 1, 0] / v_inf < 0.70


def test_discretisation_is_stable_at_large_dt():
    """零阶保持的离散无条件稳定；显式欧拉在 dt·B/M > 2 时会发散。"""
    p = AdmittanceParams(virtual_mass_kg=0.05, virtual_damping_ns_per_m=25.0,
                         max_accel_mps2=1e6)
    m = AdmittanceMapping(p, dt=0.02)      # dt·B/M = 10，显式欧拉必炸
    cmd = run(m, (20.0, 0.0), steps=500)
    assert np.all(np.isfinite(cmd)) and np.abs(cmd).max() <= p.vx_range[1] + 1e-9


# --------------------------------------------------------------------------
# 方向
# --------------------------------------------------------------------------
def test_direction_preserved_when_unsaturated():
    """各向同性增益：45° 斜拉，输出方向也该是 45°（前提是 vy 没饱和）。"""
    p = AdmittanceParams(max_accel_mps2=1e6)
    m = AdmittanceMapping(p)
    magnitude = 12.0                                  # 小到 vy 不会撞 0.5 的上限
    f = magnitude / math.sqrt(2.0)
    cmd = run(m, (f, f), steps=int(20 * p.tau_s() / DT))
    angle = math.degrees(math.atan2(cmd[-1, 1], cmd[-1, 0]))
    assert angle == pytest.approx(45.0, abs=0.5), f"方向被扭成了 {angle:.1f}°"


def test_lateral_saturates_before_forward():
    """vy 上限 0.5 比 vx 的 1.5 小，侧拉会先饱和 —— 这是物理限制，不是 bug。"""
    p = AdmittanceParams(max_accel_mps2=1e6)
    m = AdmittanceMapping(p)
    cmd = run(m, (0.0, 40.0), steps=int(20 * p.tau_s() / DT))
    assert cmd[-1, 1] == pytest.approx(p.vy_range[1], abs=1e-6)


def test_wz_always_zero():
    """牵引只影响 xy 平面，z 轴不发。"""
    m = AdmittanceMapping()
    rng = np.random.default_rng(0)
    for _ in range(200):
        f = rng.normal(0, 25.0, 3)
        assert m.step(f, 1.0)[2] == 0.0


def test_fz_does_not_affect_command():
    """Fz 估但不用。"""
    m1, m2 = AdmittanceMapping(), AdmittanceMapping()
    a = np.stack([m1.step(np.array([15.0, 3.0, +60.0]), 1.0) for _ in range(100)])
    b = np.stack([m2.step(np.array([15.0, 3.0, -60.0]), 1.0) for _ in range(100)])
    assert np.allclose(a, b)


# --------------------------------------------------------------------------
# 安全
# --------------------------------------------------------------------------
def test_clipped_to_training_distribution():
    """任何力都不能让指令超出策略的训练分布。"""
    p = AdmittanceParams()
    m = AdmittanceMapping(p)
    rng = np.random.default_rng(1)
    for _ in range(500):
        cmd = m.step(rng.normal(0, 200.0, 3), 1.0)      # 远超量程的力
        assert p.vx_range[0] - 1e-9 <= cmd[0] <= p.vx_range[1] + 1e-9
        assert p.vy_range[0] - 1e-9 <= cmd[1] <= p.vy_range[1] + 1e-9


def test_rate_limit():
    """力阶跃时，单步指令变化不超过 max_accel · dt。"""
    p = AdmittanceParams(max_accel_mps2=2.0)
    m = AdmittanceMapping(p)
    prev = np.zeros(2)
    for k in range(400):
        f = 40.0 if k % 100 < 50 else -40.0             # 来回阶跃
        cmd = m.step(np.array([f, 0.0, 0.0]), 1.0)
        assert np.linalg.norm(cmd[:2] - prev) <= p.max_accel_mps2 * DT + 1e-9
        prev = cmd[:2]


def test_confidence_gate_needs_sustained_low():
    """单帧低置信度不该停机，连续低才停 —— 否则一次抖动就把机器人闸死。"""
    p = AdmittanceParams(max_accel_mps2=1e6)
    m = AdmittanceMapping(p)
    run(m, (25.0, 0.0), steps=200)
    moving = m.state.velocity[0]
    assert moving > 0.1

    for _ in range(p.conf_gate_steps - 1):               # 差一步到闸门
        m.step(np.array([25.0, 0.0, 0.0]), p.conf_gate - 0.01)
    assert m.state.velocity[0] > 0.05, "还没到闸门步数就停了"

    for _ in range(50):                                 # 继续低 → 闸死
        m.step(np.array([25.0, 0.0, 0.0]), p.conf_gate - 0.01)
    assert m.state.velocity[0] == pytest.approx(0.0, abs=1e-3)


def test_single_low_confidence_frame_dips_but_recovers():
    """单帧置信度崩掉：允许小幅回退（置信度连续调制斜率，低置信度就该退一点），
    但不能闸死，而且要在几个时间常数内回到原值。"""
    p = AdmittanceParams(max_accel_mps2=1e6)
    m = AdmittanceMapping(p)
    run(m, (25.0, 0.0), steps=200)
    before = m.state.velocity[0]

    m.step(np.array([25.0, 0.0, 0.0]), 0.0)             # 一帧置信度 0
    assert m.state.low_conf_steps == 1, "计数器应该开始计"
    dip = (before - m.state.velocity[0]) / before
    assert 0.0 < dip < 0.20, f"单帧跌了 {dip:.0%}，太多了（滤波器没起作用？）"

    m.step(np.array([25.0, 0.0, 0.0]), 1.0)             # 置信度恢复
    assert m.state.low_conf_steps == 0, "计数器必须清零，否则偶发抖动会累积成闸门"

    run(m, (25.0, 0.0), steps=int(8 * p.tau_s() / DT))
    assert m.state.velocity[0] == pytest.approx(before, abs=1e-3), "没能恢复到原速度"


def test_confidence_scales_slope_not_min_speed():
    """置信度只调斜率。刚过死区时，任何可用置信度下的输出都要高于 stand_threshold，
    否则相位观测会在阈值附近抖（见 mapping 模块 docstring）。"""
    p = AdmittanceParams(max_accel_mps2=1e6)
    just_over = p.deadzone_on_n + 0.1
    for conf in (p.conf_gate, 0.7, 1.0):
        m = AdmittanceMapping(p)
        cmd = run(m, (just_over, 0.0), steps=int(20 * p.tau_s() / DT), conf=conf)
        assert cmd[-1, 0] > p.stand_threshold_mps, (
            f"置信度 {conf} 时起步速度 {cmd[-1, 0]:.3f} 掉进了 stand_threshold 带"
        )


def test_min_speed_clears_stand_threshold():
    p = AdmittanceParams()
    assert p.min_speed_mps > p.stand_threshold_mps
    with pytest.raises(ValueError, match="stand_threshold"):
        AdmittanceParams(min_speed_mps=0.05).validate()


# --------------------------------------------------------------------------
# 标定与状态
# --------------------------------------------------------------------------
def test_bias_calibration_cancels_offset():
    """静止时估计器有零偏，标定之后同样的零偏不该产生指令。"""
    bias = np.array([4.0, -3.0])
    m = AdmittanceMapping()
    samples = np.tile(np.array([bias[0], bias[1], 0.0]), (250, 1))
    m.calibrate_bias(samples)
    assert np.allclose(m.state.bias_xy, bias)
    cmd = run(m, (bias[0], bias[1]), steps=200)
    assert np.allclose(cmd, 0.0), "标定后零偏仍然产生了指令"


def test_reset_equals_cold_start():
    p = AdmittanceParams()
    warm = AdmittanceMapping(p)
    run(warm, (30.0, 10.0), steps=300)
    warm.reset()

    cold = AdmittanceMapping(p)
    seq = [(12.0, 0.0)] * 50 + [(0.0, 0.0)] * 50 + [(25.0, 8.0)] * 100
    a = np.stack([warm.step(np.array([x, y, 0.0]), 1.0) for x, y in seq])
    b = np.stack([cold.step(np.array([x, y, 0.0]), 1.0) for x, y in seq])
    assert np.allclose(a, b), "reset 之后的行为和冷启动不一致"


def test_reset_keeps_bias():
    m = AdmittanceMapping()
    m.calibrate_bias(np.tile(np.array([2.0, 1.0, 0.0]), (10, 1)))
    m.reset()
    assert np.allclose(m.state.bias_xy, [2.0, 1.0]), "reset 不该丢掉已标定的零偏"


# --------------------------------------------------------------------------
# 参数自洽
# --------------------------------------------------------------------------
def test_damping_derived_from_design_intent():
    """默认 B 应当正好让设计范围上限的力对应 vx 上限。"""
    p = AdmittanceParams()
    assert p.saturation_force_n() == pytest.approx(spec.FORCE_DESIGN_RANGE_N[1], abs=1e-6)
    assert p.virtual_damping_ns_per_m == pytest.approx(
        AdmittanceParams.damping_for_saturation(
            spec.FORCE_DESIGN_RANGE_N[1], p.vx_range[1], p.deadzone_off_n, p.min_speed_mps
        )
    )


def test_params_json_roundtrip(tmp_path):
    p = AdmittanceParams(virtual_mass_kg=5.0, deadzone_on_n=10.0, deadzone_off_n=7.0)
    q = AdmittanceParams.from_json(p.to_json(tmp_path / "a.json"))
    assert q == p


def test_validate_rejects_inverted_hysteresis():
    with pytest.raises(ValueError, match="迟滞"):
        AdmittanceParams(deadzone_on_n=5.0, deadzone_off_n=9.0).validate()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
