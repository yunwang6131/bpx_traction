"""板端牵引运行时的单测。

重点不是"算得准不准"（那是估计器的事），而是**出问题的时候会不会把走路搞挂**：
传感器断了、推理抛异常、操作员抢方向盘 —— 每一条都必须原样返回手柄指令。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bpx_traction" / "board"))

from bpx_traction import spec
from bpx_traction.mapping import AdmittanceParams
from traction_runtime import (MAX_BAD_FRAMES, FrameRing, TractionRuntime,
                              build_traction_frame)

ONNX = spec.DEFAULT_ESTIMATOR_DIR / "estimator.onnx"
Q0 = np.asarray(spec.DEFAULT_JOINT_POS, dtype=np.float32)
#: 测运行时机制的用例用它：验的是"坏了会不会把走路带下水"，和死区标没标定过无关。
#: 真实的默认参数是 calibrated=False（现有估计器定不出可用死区），
#: 那条由 test_preflight_refuses_uncalibrated_params 单独守。
CALIBRATED = AdmittanceParams(calibrated=True)


def gravity_xyzw(quat):
    """和板子 `projected_gravity_xyzw` 等价的参考实现，只给测试用。"""
    x, y, z, w = (float(v) for v in quat)
    return np.array([2.0 * (x * z - w * y),
                     2.0 * (y * z + w * x),
                     1.0 - 2.0 * (x * x + y * y)], dtype=np.float32)


def snapshot(tau=True, acc=True, standing=True):
    rng = np.random.default_rng(0)
    q = Q0 + (0.0 if standing else rng.normal(0, 0.05, 12).astype(np.float32))
    return {
        'q': q.astype(np.float32),
        'dq': np.zeros(12, np.float32),
        'omega': np.zeros(3, np.float32),
        'quat': np.array([0.0, 0.0, 0.0, 1.0], np.float32),      # xyzw，单位四元数
        'tau': np.full(12, 4.0, np.float32) if tau else np.full(12, np.nan),
        'acc': np.array([0.0, 0.0, 9.81], np.float32) if acc else np.full(3, np.nan),
    }


# --------------------------------------------------------------------------
# 帧组装
# --------------------------------------------------------------------------
def test_frame_layout_matches_contract():
    frame, ok = build_traction_frame(snapshot(), np.zeros(12), gravity_xyzw)
    assert ok and frame.shape == (spec.FRAME_SIZE,)
    g = lambda n: frame[slice(*spec.term_slice(n))]
    assert np.allclose(g("imu_acc_b"), [0, 0, 9.81])
    assert np.allclose(g("joint_pos_rel"), 0.0)          # 站在 q0 上
    assert np.allclose(g("joint_torque"), 4.0)
    # 单位四元数在板子的 xyzw 下是 [0,0,0,1]，契约 wxyz 要变成 [1,0,0,0]
    assert np.allclose(g("quat_wb"), [1, 0, 0, 0])
    assert np.allclose(g("projected_gravity_b"), [0, 0, 1])


def test_torque_falls_back_to_pd_when_getter_dies():
    """力矩 getter 挂了要用 PD 反推填上，而不是喂零 —— 它是最重要的输入。"""
    snap = snapshot(tau=False)
    snap['q'] = Q0 - 0.1                                  # 偏离目标 0.1 rad
    target = Q0
    frame, ok = build_traction_frame(snap, np.zeros(12), gravity_xyzw,
                                     policy_target=target, kp=40.0, kd=1.0)
    assert ok, "只有力矩读不到时应该还能用（加速度还在）"
    tau = frame[slice(*spec.term_slice("joint_torque"))]
    assert np.allclose(tau, 40.0 * 0.1), f"PD 反推该是 kp·Δq = 4.0，实际 {tau[:3]}"


def test_missing_accel_is_not_recoverable():
    """加速度没有退路 —— 旧数据里推不出来，真机上也推不出来，只能判这帧不可信。"""
    _frame, ok = build_traction_frame(snapshot(acc=False), np.zeros(12), gravity_xyzw,
                                      policy_target=Q0)
    assert not ok


# --------------------------------------------------------------------------
# 环形缓冲
# --------------------------------------------------------------------------
def test_ring_backfills_on_first_push():
    """第一帧要把 25 个槽位填满，不能让估计器看到一窗零。"""
    ring = FrameRing()
    f = np.arange(spec.FRAME_SIZE, dtype=np.float32)
    buf = ring.push(f)
    assert buf.shape == (spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    assert np.allclose(buf, f)


def test_ring_keeps_newest_last():
    ring = FrameRing()
    for k in range(spec.HISTORY_LENGTH + 10):
        buf = ring.push(np.full(spec.FRAME_SIZE, k, dtype=np.float32))
    assert buf[-1][0] == pytest.approx(spec.HISTORY_LENGTH + 9)
    assert buf[0][0] == pytest.approx(10)
    assert np.all(np.diff(buf[:, 0]) == 1), "历史必须是连续递增的，最新在最后"


# --------------------------------------------------------------------------
# 运行时：失败路径
# --------------------------------------------------------------------------
pytestmark_onnx = pytest.mark.skipif(not ONNX.is_file(), reason=f"没有 {ONNX}")


class _FakeIO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _FakeSession:
    def __init__(self, ins, outs):
        self._ins, self._outs = ins, outs

    def get_inputs(self):
        return self._ins

    def get_outputs(self):
        return self._outs


def _check(ins, outs):
    """只跑签名校验，不碰 onnxruntime。"""
    rt = TractionRuntime.__new__(TractionRuntime)
    rt.session = _FakeSession(ins, outs)
    rt._check_signature()


def test_signature_check_accepts_the_real_contract():
    _check([_FakeIO('sensor_history', ['batch', spec.HISTORY_LENGTH, spec.FRAME_SIZE])],
           [_FakeIO('values', ['batch', spec.OUTPUT_SIZE]), _FakeIO('confidence', ['batch', 4])])


@pytest.mark.parametrize("ins,outs,needle", [
    # 喂错了帧宽——比如板子适配器少拼了一段
    ([_FakeIO('sensor_history', ['batch', 25, 45])],
     [_FakeIO('values', ['batch', 15]), _FakeIO('confidence', ['batch', 4])], '契约要'),
    # 历史长度不对
    ([_FakeIO('sensor_history', ['batch', 10, 61])],
     [_FakeIO('values', ['batch', 15]), _FakeIO('confidence', ['batch', 4])], '契约要'),
    # 拿走路策略的 onnx 当估计器喂进来（两个输入）
    ([_FakeIO('policy', ['batch', 45]), _FakeIO('adaptation_history', ['batch', 1125])],
     [_FakeIO('actions', ['batch', 12])], '只有一个输入'),
    # 输出头对不上
    ([_FakeIO('sensor_history', ['batch', 25, 61])],
     [_FakeIO('force', ['batch', 3])], 'values'),
])
def test_signature_check_rejects_the_wrong_model(ins, outs, needle):
    """形状对不上要在构造时就拒掉，不能等跑起来才炸 —— 那时机器人已经站着了。"""
    with pytest.raises(ValueError, match=needle):
        _check(ins, outs)


@pytestmark_onnx
def test_disabled_runtime_is_a_passthrough():
    rt = TractionRuntime(ONNX, gravity_xyzw)
    pad = np.array([0.3, 0.0, 0.5], np.float32)
    assert np.allclose(rt.fuse(pad, snapshot(), np.zeros(12)), pad)


@pytestmark_onnx
def test_preflight_refuses_uncalibrated_params():
    """死区没标定过就不许启用。这是现在挡住牵引上真机的那道闸，别拆。"""
    rt = TractionRuntime(ONNX, gravity_xyzw, params=AdmittanceParams(calibrated=False))
    ok, why = rt.preflight(snapshot(), np.zeros(12))
    assert not ok and "calibrated" in why


@pytestmark_onnx
def test_preflight_refuses_when_standing_force_already_exceeds_deadzone():
    """启用前估一帧：静止时估出来的力要是已经越过起步阈值，一开牵引机器人就自己走。

    这条正是现有估计器的处境 —— 合成的静止帧上它报 24 N，而占位死区是 8 N。
    """
    rt = TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED)
    ok, why = rt.preflight(snapshot(), np.zeros(12), policy_target=Q0)
    assert not ok and "自己走" in why, why


def _force_enabled(rt):
    """把运行时直接置入"已启用"状态，跳过 preflight。

    下面几条验的是 `fuse()` 的失效和覆盖行为，和"估计器准不准"无关。走 `enable()`
    的话会被 preflight 挡住（它本来就该挡 —— 见上一条），测不到想测的东西。
    """
    rt.enabled = True
    rt.mapping.reset()
    rt.ring.filled = False
    return rt


@pytestmark_onnx
def test_inference_blowup_falls_back_to_pad_and_disables():
    """推理抛异常时必须原样返回手柄指令并自己关掉 —— 不能把走路带下水。"""
    rt = _force_enabled(TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED))
    rt.session = None                                     # 模拟推理链路炸掉
    pad = np.array([0.0, 0.0, 0.0], np.float32)
    assert np.allclose(rt.fuse(pad, snapshot(), np.zeros(12)), pad)
    assert not rt.enabled, "出异常之后牵引必须自己关掉"


@pytestmark_onnx
def test_dead_sensors_disable_after_a_full_window():
    rt = _force_enabled(TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED))
    pad = np.zeros(3, np.float32)
    for _ in range(MAX_BAD_FRAMES - 1):
        rt.fuse(pad, snapshot(acc=False), np.zeros(12))
        assert rt.enabled, "单帧丢数不该关掉牵引"
    rt.fuse(pad, snapshot(acc=False), np.zeros(12))
    assert not rt.enabled


# --------------------------------------------------------------------------
# 运行时：融合规则
# --------------------------------------------------------------------------
@pytestmark_onnx
def test_stick_always_wins():
    """操作员一碰摇杆，牵引立刻让位 —— 这是唯一的人工覆盖手段，不能有例外。

    关键在于这条**即使估计器在乱报也成立**：合成帧上它报 24 N，远超死区，
    牵引本来会输出一个不小的速度；摇杆一动就必须全部让位。
    """
    rt = _force_enabled(TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED))
    pad = np.array([0.8, -0.2, 0.3], np.float32)
    for _ in range(3 * spec.HISTORY_LENGTH):
        out = rt.fuse(pad, snapshot(standing=False), np.zeros(12), policy_target=Q0)
    assert np.allclose(out, pad)
    assert rt.last_force is not None and np.linalg.norm(rt.last_force[:2]) > 0, (
        "摇杆优先的同时，估计器和映射的状态仍要照常推进，否则一松手是冷的")


@pytestmark_onnx
def test_yaw_always_comes_from_the_stick():
    """牵引只管 xy 平面，wz 必须原样透传 —— 被侧拉时机器人保持朝向。"""
    rt = _force_enabled(TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED))
    for _ in range(3 * spec.HISTORY_LENGTH):
        out = rt.fuse(np.zeros(3, np.float32), snapshot(), np.zeros(12), policy_target=Q0)
    assert out.shape == (3,)
    assert out[2] == 0.0, "摇杆 wz 为零时输出的 wz 也必须是零"


# --------------------------------------------------------------------------
# 真机回放：合成快照证明不了什么，这条才算数
# --------------------------------------------------------------------------
REAL = spec.TRACTION_LOG_DIR.parent / "real_v6" / "rec_20260418_191052"


#: 真机无外力录制上，整条链路（估计器 → 导纳）当前的误起步占比。
#:
#: **这是已知缺陷的记录，不是规格。** 那段录制全程没人碰机器人，正确答案是
#: 全程零指令；现有估计器误报 p99 39–46 N，而设计力范围上限只有 30 N，
#: 所以不存在能挡住它的死区（见 `mapping.py` 的 `deadzone_on_n` 注释）。
#:
#: 这条测试是**棘轮**：估计器改好之后把这两个数往下收，别让它悄悄退化回来。
#: 目标是 0。
REAL_FALSE_ENGAGE_PCT = 96.0
REAL_FALSE_PEAK_MPS = 1.3


@pytest.mark.skipif(not ONNX.is_file() or not (REAL.parent / (REAL.name + "_joint500.bin")).is_file(),
                    reason="没有真机录制或估计器")
def test_real_recording_false_engagement_does_not_regress():
    """把真机 300 s 无外力行走逐帧喂进整条链路，量误起步率。

    和 `real_deadzone.py` 的区别：那边只看力的分布，这边跑的是**真正会上板的
    代码路径** —— 环形缓冲、ONNX 会话、Schmitt 迟滞、去抖、一阶导纳、速率限幅
    全在里面。链路里任何一环改坏了，这里会红。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from real_deadzone import build_frames

    frames, _cmd, usable = build_frames(REAL)
    rt = TractionRuntime(ONNX, gravity_xyzw, params=CALIBRATED)
    rt.enabled = True                       # 直接喂帧，跳过 preflight 的快照组装
    rt.mapping.reset()

    engaged, total, peak = 0, 0, 0.0
    for k in range(spec.HISTORY_LENGTH, len(frames)):
        if not usable[k - spec.HISTORY_LENGTH + 1:k + 1].all():
            rt.ring.filled = False          # 跨过策略重启断点，历史重新灌
            continue
        force, conf = rt._infer(rt.ring.push(frames[k]))
        speed = float(np.linalg.norm(rt.mapping.step(force, conf)[:2]))
        peak = max(peak, speed)
        engaged += speed > 0.0
        total += 1

    pct = 100.0 * engaged / max(1, total)
    print(f"\n真机无外力录制：误起步 {pct:.2f}%（{engaged}/{total}），峰值 {peak:.3f} m/s")
    assert pct <= REAL_FALSE_ENGAGE_PCT, (
        f"误起步率 {pct:.2f}% 比记录的 {REAL_FALSE_ENGAGE_PCT}% 还差，整条链路退化了")
    assert peak <= REAL_FALSE_PEAK_MPS, (
        f"误起步峰值 {peak:.3f} m/s 比记录的 {REAL_FALSE_PEAK_MPS} 还高")
