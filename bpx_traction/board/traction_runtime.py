#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端牵引运行时：把机器人状态拼成 61 维帧、跑力估计器、经导纳映射出速度指令。

**只依赖 numpy + onnxruntime**（板子上的版本：numpy 1.21.5 / onnxruntime 1.23.2，
Python 3.10，CPU）。不碰 torch、不碰 isaaclab。

设计约束，按重要性排：

1. **牵引失败绝不能影响走路。** 任何一步出异常，`fuse()` 原样返回手柄指令，
   并把牵引置为关闭。策略那条链路一个字都不改。
2. **手柄永远优先。** 操作员一碰摇杆，牵引立刻让位；摇杆归中才轮到牵引。
   这是最可预测的覆盖方式，不需要操作员记任何规则。
3. **wz 永远来自手柄。** 牵引只影响 xy 平面（`AdmittanceMapping` 的第三维恒 0）。

和开发机共用同一份契约（`bpx_traction.spec`）和同一份映射（`bpx_traction.mapping`），
不复制常数 —— 复制出来的常数迟早会和训练那边对不上。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

_PKG = Path(__file__).resolve().parents[1]           # .../bpx_traction
if str(_PKG.parent) not in sys.path:
    sys.path.insert(0, str(_PKG.parent))

from bpx_traction import spec                                    # noqa: E402
from bpx_traction.mapping import AdmittanceMapping, AdmittanceParams  # noqa: E402

#: 摇杆任一轴超过它就算"操作员在控"，牵引让位。手柄静区之上一点点。
STICK_ACTIVE = 0.05
#: 连续多少帧拿不到有效的 tau/acc 就关掉牵引。50 Hz 下 25 帧 = 0.5 s = 一个历史窗。
MAX_BAD_FRAMES = 25


class FrameRing:
    """25 × 61 的环形历史，最新帧在最后 —— 和 `spec.HISTORY_NEWEST_LAST` 一致。

    复位时用当前帧填满所有槽位（和板子策略 `prepare_policy_state` 的做法相同），
    这样刚启用牵引的那一瞬间不会喂给估计器一窗零。
    """

    def __init__(self) -> None:
        self.buf = np.zeros((spec.HISTORY_LENGTH, spec.FRAME_SIZE), dtype=np.float32)
        self.filled = False

    def reset(self, frame: np.ndarray) -> None:
        self.buf[:] = frame
        self.filled = True

    def push(self, frame: np.ndarray) -> np.ndarray:
        if not self.filled:
            self.reset(frame)
        else:
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = frame
        return self.buf


def build_traction_frame(snap, last_action, projected_gravity, policy_target=None,
                         kp=40.0, kd=1.0):
    """板子的状态快照 → 61 维估计器输入帧。

    Args:
        snap: `read_high_snapshot()` 的返回，要有 q / dq / omega / quat / tau / acc。
        last_action: 上一步策略的**原始**网络输出 (12,)，和 45 维观测里那一段同源。
        projected_gravity: 板子自己的 `projected_gravity_xyzw` 函数 —— 传进来而不是
            在这里重写，避免两处实现漂移。
        policy_target: 当前策略目标 (12,)。只在 `snap['tau']` 读不到时用来反推力矩。
        kp, kd: 反推力矩用的增益，要和板子 POLICY 档的 KP/KD 一致。

    Returns:
        (frame (61,) float32, ok: bool)。``ok`` 为假表示这一帧的传感器不可信。
    """
    frame = np.zeros(spec.FRAME_SIZE, dtype=np.float32)
    put = lambda name, v: frame.__setitem__(slice(*spec.term_slice(name)), v)

    q, dq = snap['q'], snap['dq']
    put("imu_gyro_b", snap['omega'])
    put("projected_gravity_b", projected_gravity(snap['quat']))
    # 板子的 IMU 返回 xyzw（真机实测三次确认，手册写的 wxyz 是错的），契约要 wxyz。
    quat = np.asarray(snap['quat'], dtype=np.float32)
    put("quat_wb", quat[[3, 0, 1, 2]] if spec.QUAT_ORDER_ON_BOARD == "xyzw" else quat)
    put("joint_pos_rel", q - np.asarray(spec.DEFAULT_JOINT_POS, dtype=np.float32))
    put("joint_vel", dq)
    put("last_action", last_action)

    acc = np.asarray(snap.get('acc'), dtype=np.float32)
    ok = acc.shape == (3,) and np.isfinite(acc).all()
    put("imu_acc_b", acc if ok else 0.0)

    tau = np.asarray(snap.get('tau'), dtype=np.float32)
    if tau.shape == (12,) and np.isfinite(tau).all():
        put("joint_torque", tau)
    elif policy_target is not None:
        # 退路：实测力矩 = PD 指令力矩滞后 8 ms、残差 0.085 N·m（真机标定，占 2.3%）。
        # 读不到 getter 时用这个，比喂零或 NaN 强得多 —— 力矩是估计器第一重要的输入
        # （消融：置零 xy 误差 +351%）。
        put("joint_torque", kp * (np.asarray(policy_target, np.float32) - q) - kd * dq)
    else:
        ok = False

    return frame, bool(ok)


class TractionRuntime:
    """力估计器 + 导纳映射，封装成一个 `fuse(手柄指令, …) -> 实际指令`。

    对外只有三个方法：`fuse()` 每个策略步调一次，`enable()/disable()` 开关。
    出任何问题都自己关掉并打印原因，不往上抛。
    """

    def __init__(self, onnx_path, projected_gravity, params=None,
                 kp=40.0, kd=1.0, fusion='preempt', log=print):
        import onnxruntime as ort

        if fusion not in ('preempt', 'add'):
            raise ValueError("fusion 只能是 'preempt'（手柄一动牵引全让位，默认、最安全）"
                             "或 'add'（牵引与手柄 xy 叠加，手感连续但误差也叠加）")
        self.fusion = fusion
        self.log = log
        self.gravity = projected_gravity
        self.kp, self.kd = kp, kd
        self.params = params or AdmittanceParams.from_json()
        self.mapping = AdmittanceMapping(self.params, dt=1.0 / spec.POLICY_HZ)
        self.ring = FrameRing()
        self.enabled = False
        self.bad_frames = 0
        self.last_force = np.zeros(3, dtype=np.float32)
        self.last_conf = 0.0

        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self._check_signature()

    # ------------------------------------------------------------------ 门4
    def _check_signature(self) -> None:
        """ONNX 的形状必须和契约对上。对不上就不是这个估计器，直接拒。"""
        ins = self.session.get_inputs()
        if len(ins) != 1:
            raise ValueError('估计器该只有一个输入，实际 %d 个：%s'
                             % (len(ins), [i.name for i in ins]))
        self.input_name = ins[0].name
        want = [spec.HISTORY_LENGTH, spec.FRAME_SIZE]
        got = list(ins[0].shape[1:])
        if got != want:
            raise ValueError('估计器输入 %s 是 %s，契约要 [batch, %d, %d]'
                             % (self.input_name, ins[0].shape, *want))
        outs = [o.name for o in self.session.get_outputs()]
        if outs[:2] != ['values', 'confidence']:
            raise ValueError('估计器输出该是 (values, confidence)，实际 %s' % outs)
        ov = list(self.session.get_outputs()[0].shape[1:])
        if ov != [spec.OUTPUT_SIZE]:
            raise ValueError('values 维度 %s，契约要 %d' % (ov, spec.OUTPUT_SIZE))

    def preflight(self, snap, last_action, policy_target=None):
        """启用前的自检。返回 ``(ok, 原因)``，不改变状态。

        四件事：参数标定过没有、传感器在不在、推理跑不跑得通、
        **静止时估计出来的力有没有超过起步阈值**（最后这条最要紧 ——
        没过的话一开牵引机器人就自己走了）。
        """
        if not self.params.calibrated:
            return False, '导纳参数没有真机标定过（calibrated=False）'
        frame, ok = build_traction_frame(snap, last_action, self.gravity,
                                         policy_target, self.kp, self.kd)
        if not ok:
            return False, '快照里 tau/acc 读不到（加速度没有退路，必须有）'
        try:
            force, conf = self._infer(np.repeat(frame.reshape(1, -1),
                                                spec.HISTORY_LENGTH, axis=0))
        except Exception as exc:
            return False, '首帧推理失败: %r' % (exc,)
        magnitude = float(np.linalg.norm(force[:2]))
        if magnitude >= self.params.deadzone_on_n:
            return False, ('静止估计力 %.1f N 已经越过起步阈值 %.1f N，'
                           '现在启用机器人会自己走'
                           % (magnitude, self.params.deadzone_on_n))
        return True, '静止估计力 %.1f N / 起步阈值 %.1f N，置信度 %.2f' % (
            magnitude, self.params.deadzone_on_n, conf)

    # ------------------------------------------------------------------ 运行
    def enable(self, snap, last_action, policy_target=None) -> tuple[bool, str]:
        ok, why = self.preflight(snap, last_action, policy_target)
        if not ok:
            return False, why
        self.mapping.reset()
        self.ring.filled = False
        self.bad_frames = 0
        self.enabled = True
        return True, why

    def disable(self, reason: str) -> None:
        if self.enabled:
            self.log('[牵引] 关闭：%s' % reason)
        self.enabled = False
        self.mapping.reset()
        self.ring.filled = False

    def _infer(self, history: np.ndarray):
        out = self.session.run(None, {self.input_name: history[None].astype(np.float32)})
        values, confidence = out[0][0], out[1][0]
        force = values[slice(*spec.output_slice("force_b"))]
        conf = float(confidence[spec.CONFIDENCE_TERMS.index("force")])
        return force, conf

    def fuse(self, pad_cmd, snap, last_action, policy_target=None) -> np.ndarray:
        """每个 50 Hz 策略步调一次。返回真正要发给策略的 (vx, vy, wz)。

        牵引没开、出错、或者操作员在动摇杆时，原样返回 `pad_cmd`。
        """
        pad = np.asarray(pad_cmd, dtype=np.float32).reshape(3)
        if not self.enabled:
            return pad
        try:
            frame, ok = build_traction_frame(snap, last_action, self.gravity,
                                             policy_target, self.kp, self.kd)
            if not ok:
                self.bad_frames += 1
                if self.bad_frames >= MAX_BAD_FRAMES:
                    self.disable('连续 %d 帧读不到 tau/acc' % self.bad_frames)
                return pad
            self.bad_frames = 0
            history = self.ring.push(frame)

            # 历史和映射状态每步都推进（否则一松手映射是冷的）。
            force, conf = self._infer(history)
            self.last_force, self.last_conf = force, conf
            traction = self.mapping.step(force, conf)

            # wz 永远来自手柄 —— 牵引只管 xy 平面，这条两种模式都成立。
            if self.fusion == 'add':
                # 叠加：牵引 xy + 手柄 xy，和夹到**训练分布**（不是板子限幅），
                # 否则超出策略见过的范围会被策略端静默截断。死区已挡住低力误报，
                # 所以静区内叠加的只有确信的 >deadzone 拉力。
                vx = _clip(traction[0] + pad[0], *self.params.vx_range)
                vy = _clip(traction[1] + pad[1], *self.params.vy_range)
                return np.array([vx, vy, pad[2]], dtype=np.float32)
            # preempt（默认）：手柄一动，牵引 xy 全让位。
            if float(np.abs(pad).max()) > STICK_ACTIVE:
                return pad
            return np.array([traction[0], traction[1], pad[2]], dtype=np.float32)
        except Exception as exc:
            self.disable('运行异常 %r' % (exc,))
            return pad

    def status(self) -> str:
        return '%s[%s] |F_xy| %5.1f N  conf %.2f' % (
            '开' if self.enabled else '关', self.fusion,
            float(np.linalg.norm(self.last_force[:2])), self.last_conf)
