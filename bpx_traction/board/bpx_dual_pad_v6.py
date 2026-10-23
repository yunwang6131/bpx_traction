#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BPX unified controller: mutually-exclusive MOTION and JOINT/POLICY layers.

The operator workflow is intentionally kept compatible with the original dual-pad
program:

MOTION (default): Y stand, A sit, X/RS damping, LB walk, RB run; while lying,
RB switches to JOINT.
JOINT: Y joint-PD stand then auto-start policy after all safety gates pass;
A joint-PD sit, X/RS damping, RB manually starts policy as a fallback, and LB
returns to MOTION only after sitting and damping.
POLICY: sticks command velocity, LT/RT select gear, A stops policy and sits,
X/RS damping.

Only the 500 Hz joint command thread calls setJointCommand().  The 50 Hz policy
thread only publishes a checked target snapshot.
"""
import argparse
import csv
import gc
import json
import math
import os
import queue
import re
import select
import signal
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np

try:
    import bpx_sdk
except ImportError as exc:
    print('[FATAL] 未安装 bpx_sdk: %r' % (exc,), flush=True)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent
POLICY_ONNX = ROOT / 'deploy/robots/bpx/config/policy/velocity_rma/v0/exported/policy.onnx'
POLICY_CFG = ROOT / 'deploy/robots/bpx/config/policy/velocity_rma/v0/params/deploy.yaml'
LOG_DIR = ROOT / 'log'

# Joint/model constants.  Measured SDK poses are deliberately distinct from
# policy/MJCF limits: the policy stays disarmed until the convention gate is true.
LT, LC = 0.23, 0.2315648
F_FRONT, F_HIND = 31.2, 62.1
# BPX velocity-RMA policy/MJCF order: FL, FR, HL, HR x roll, pitch, knee.
# Its documented startup contract requires the PD stand target to equal q0.
DEFAULT_POSE = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float32)
STAND_POSE = DEFAULT_POSE.copy()
LIE_POSE = np.array([
    -0.0448, 1.1757, -2.7683, 0.0224, 1.1439, -2.7781,
     0.0004, 1.1701, -2.7739, -0.0084, 1.1971, -2.7798,
], dtype=np.float32)
MODEL_LIMITS = np.array([
    (-0.8727, 0.8727), (-0.7854, 2.60661), (-2.7531, -0.5539),
] * 4, dtype=np.float32)
KP_PD, KD_PD = 45.0, 1.5
KP_POLICY, KD_POLICY = 40.0, 1.0
KD_DAMP = 3.0
EFFORT_LIMIT = 30.0
ACTION_SCALE = 0.1875
ACTION_CLIP = 4.0
HZ_JOINT, HZ_POLICY = 500.0, 50.0
POLICY_FRAME_SIZE, POLICY_HISTORY_LENGTH = 45, 25
POLICY_HISTORY_INPUT_SIZE = POLICY_FRAME_SIZE * POLICY_HISTORY_LENGTH
# Frame layout: joint_pos, joint_vel, base_ang_vel,
# projected_gravity, last_action, command. History flattening is term-major.
POLICY_TERM_SLICES = ((0, 12), (12, 24), (24, 27), (27, 30),
                      (30, 42), (42, 45))
TS_PD, SLEW, RELEASE = 3.0, 2.5, 1.2
STATE_TIMEOUT = 0.10
POLICY_TARGET_TIMEOUT = 0.10
MAX_SEND_FAILURES = 5
MOTION_RECONNECT_FAILURES = 10   # ~100 ms at the 100 Hz main loop
JOINT_RECONNECT_FAILURES = 50     # ~100 ms at the 500 Hz joint loop

# This bpx_sdk build keeps its native high-rate UDP receiver alive after
# disconnect() while the Python process continues. Reusing 7895 therefore
# breaks the second JOINT entry with EADDRINUSE. Give every Joint client a
# new locally-free port and tell the robot that port via setJointStateUploadPort.
JOINT_UDP_PORT_MIN = 17895
JOINT_UDP_PORT_MAX = 18150

MOTION_CAPS = {'vx': 2.0, 'vy': 0.6, 'yaw': 1.5}
MOTION_GAIT_LIM = {'walk': (1.5, 0.5, 2.0), 'running': (3.0, 1.0, 2.0)}
GEARS = [0.25, 0.50, 0.75, 1.00]
# Exact play command bounds from the supplied training configuration.
POLICY_VX_BACK_MAX, POLICY_VX_FWD_MAX = 6.0, 6.0
POLICY_VY_MAX, POLICY_YAW_MAX = 0.5, 2.5

EVENT_FMT, EVENT_SIZE = '<qqHHi', struct.calcsize('<qqHHi')
BTN_NAME = {
    0x130: 'A', 0x131: 'B', 0x133: 'X', 0x134: 'Y',
    0x136: 'LB', 0x137: 'RB', 0x13A: 'BACK', 0x13B: 'START',
    0x13D: 'LS', 0x13E: 'RS',
}


def mono():
    return time.monotonic()


def vec(value, size, label):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < size:
        raise ValueError('%s 长度=%d，期望至少 %d' % (label, arr.size, size))
    arr = arr[:size].copy()
    if not np.isfinite(arr).all():
        raise ValueError('%s 含 NaN/Inf' % label)
    return arr


def smoothstep(value):
    value = max(0.0, min(1.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def gravity_tff(q):
    out = np.zeros(12, dtype=np.float32)
    for leg in range(4):
        hip, knee = float(q[leg * 3 + 1]), float(q[leg * 3 + 2])
        load = F_FRONT if leg < 2 else F_HIND
        knee_arm = -LC * math.sin(hip + knee)
        hip_arm = -LT * math.sin(hip) + knee_arm
        out[leg * 3 + 1] = load * hip_arm
        out[leg * 3 + 2] = load * knee_arm
    return out


def projected_gravity_xyzw(quat):
    """R(body->world)^T * [0,0,-1], SDK-header order x,y,z,w."""
    x, y, z, w = (float(v) for v in vec(quat, 4, 'imu_quat'))
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not (0.5 <= norm <= 1.5):
        raise ValueError('四元数模长异常 %.4f' % norm)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([
        2.0 * (w*y - x*z),
       -2.0 * (w*x + y*z),
       -(1.0 - 2.0 * (x*x + y*y)),
    ], dtype=np.float32)


def find_pad():
    try:
        text = Path('/proc/bus/input/devices').read_text(errors='replace')
    except OSError:
        return None, None
    keywords = ('x-box', 'xbox', 'pad', 'gamepad', 'betop', 'asura')
    for block in text.split('\n\n'):
        name_m = re.search(r'Name="([^"]*)"', block)
        handlers_m = re.search(r'Handlers=([^\n]+)', block)
        if not (name_m and handlers_m):
            continue
        if not any(word in name_m.group(1).lower() for word in keywords):
            continue
        for token in handlers_m.group(1).split():
            if token.startswith('event'):
                return '/dev/input/' + token, name_m.group(1)
    return None, None


class Pad:
    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.press = []
        self.btn = {}
        self.axes = {}
        self.online = True
        self.last_event = mono()

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def mark_offline(self, reason):
        if self.online:
            print('[PAD-FAULT] %s；速度清零并进入安全状态' % reason, flush=True)
        self.online = False
        self.axes.clear()
        self.btn.clear()
        self.press.clear()

    def update(self):
        if not self.online or self.fd is None or not os.path.exists(self.path):
            self.mark_offline('手柄设备消失')
            return False
        while True:
            try:
                readable, _, _ = select.select([self.fd], [], [], 0)
            except (OSError, ValueError) as exc:
                self.mark_offline('手柄 select 失败: %r' % (exc,))
                return False
            if not readable:
                return True
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                return True
            except OSError as exc:
                self.mark_offline('手柄读取失败: %r' % (exc,))
                return False
            if not chunk:
                self.mark_offline('手柄 EOF/已拔出')
                return False
            self.last_event = mono()
            for off in range(0, len(chunk) - EVENT_SIZE + 1, EVENT_SIZE):
                _, _, typ, code, value = struct.unpack_from(EVENT_FMT, chunk, off)
                if typ == 1:
                    if value and not self.btn.get(code):
                        self.press.append(code)
                    self.btn[code] = value
                elif typ == 3:
                    self.axes[code] = value

    def pop(self):
        return self.press.pop(0) if self.press else None

    def norm(self, code, lo, hi):
        if not self.online:
            return 0.0
        value = self.axes.get(code, 0)
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        scaled = (value - center) / half if half else 0.0
        return 0.0 if abs(scaled) < 0.12 else max(-1.0, min(1.0, scaled))


# ---------------- walk recording (for sim replay / closed-loop comparison) ----------------
# 500 Hz: each tick, the state read + the command actually sent (kp, q_cmd, kd, dq_cmd, tff).
# 50 Hz: each inference, velocity command / 45-d observation / raw action / joint target.
# Files: log/rec_<time>_joint500.bin, log/rec_<time>_policy50.bin; column names in the .json beside each:
#   cols = json.load(open(f[:-4] + '.json'))['columns']
#   data = np.fromfile(f, dtype='<f8').reshape(-1, len(cols))
# Rows only in JOINT/POLICY. Recorder errors are only counted, never affect control. --no-record = off.
_RJ = ['%s_%s' % (l, j) for l in ('fl', 'fr', 'hl', 'hr') for j in ('roll', 'pitch', 'knee')]
REC_JOINT_COLS = (['t', 'robot_ts', 'seq', 'state_age', 'mode', 'sub', 'send_ok']
                  + ['q_' + n for n in _RJ] + ['dq_' + n for n in _RJ]
                  + ['quat_x', 'quat_y', 'quat_z', 'quat_w', 'omega_x', 'omega_y', 'omega_z']
                  + ['kp_' + n for n in _RJ] + ['qcmd_' + n for n in _RJ] + ['kd_' + n for n in _RJ]
                  + ['dqcmd_' + n for n in _RJ] + ['tff_' + n for n in _RJ]
                  + ['cmd_vx', 'cmd_vy', 'cmd_wz'] + ['ptarget_' + n for n in _RJ]
                  # --- v6 牵引数据采集：力估计器 61 维输入里真机没验过的那 15 维 ---
                  + ['tau_' + n for n in _RJ] + ['acc_x', 'acc_y', 'acc_z'] + ['snap_us'])
RECORD_EXTRA_SENSORS = True   # 翻 False 跑一段，和 True 那段的 snap_us 相减 = 两个新 getter 的净开销
REC_POLICY_COLS = (['t', 'step', 'session', 'cmd_vx', 'cmd_vy', 'cmd_wz']
                   + ['obs%02d' % i for i in range(45)] + ['raw_' + n for n in _RJ] + ['target_' + n for n in _RJ])
REC_MODE = {'connecting': 0, 'motion': 1, 'transition': 2, 'joint': 3, 'policy': 4, 'reconnecting': 5, 'fault': 6}
REC_SUB = {'damp': 0, 'pd': 1, 'policy': 2, 'release': 3}
REC_NAN3, REC_NAN12 = np.full(3, np.nan), np.full(12, np.nan)


class _RecWriter:
    """Fixed-width float64 rows. add() only copies to memory; a background thread writes to disk."""

    def __init__(self, path, cols, block=2500):
        self.path, self.cols, self.block = path, cols, block
        self.buf, self.n, self.rows, self.lost = np.empty((block, len(cols))), 0, 0, 0
        path.with_suffix('.json').write_text(json.dumps(
            {'columns': cols, 'dtype': '<f8', 'mode': REC_MODE, 'sub': REC_SUB}, indent=1))
        self.fh = open(path, 'wb')
        self.q = queue.Queue(maxsize=32)
        self.th = threading.Thread(target=self._run, name='bpx-recorder', daemon=True)
        self.th.start()

    def add(self, row):
        self.buf[self.n] = row
        self.n += 1
        self.rows += 1
        if self.n == self.block:
            try:
                self.q.put_nowait(self.buf)
            except queue.Full:
                self.lost += self.n
            self.buf, self.n = np.empty_like(self.buf), 0

    def _run(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            try:
                self.fh.write(b.astype('<f8').tobytes())
                self.fh.flush()
            except Exception:
                self.lost += len(b)

    def close(self):
        try:
            if self.n:
                self.q.put(self.buf[:self.n].copy(), timeout=1.0)
            self.q.put(None, timeout=1.0)
        except queue.Full:
            pass
        self.th.join(timeout=5.0)
        self.fh.close()
        if self.rows == 0:  # never entered JOINT/POLICY: leave no empty files
            for p in (self.path, self.path.with_suffix('.json')):
                try:
                    p.unlink()
                except OSError:
                    pass
        else:
            print('[rec] %s: %d rows, %d lost' % (self.path.name, self.rows, self.lost), flush=True)


class WalkRecorder:
    def __init__(self, log_dir):
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        self.t0 = mono()
        self.jw = _RecWriter(log_dir / ('rec_%s_joint500.bin' % stamp), REC_JOINT_COLS)
        self.pw = _RecWriter(log_dir / ('rec_%s_policy50.bin' % stamp), REC_POLICY_COLS)
        self.cmd, self.target = REC_NAN3, REC_NAN12
        self.step = self.session = 0
        self.errors = 0
        self.closed = False
        print('[rec] -> %s/rec_%s_*.bin' % (log_dir, stamp), flush=True)

    def joint(self, app, now, snap, command, ok):
        """500 Hz thread: snap = state read this tick, command = what was sent (kp, q, kd, dq, tff)."""
        if self.closed:
            return
        try:
            with app.state_lock:
                sub = app.S['sub']
            try:
                ts = float(snap.get('robot_ts'))
            except Exception:
                ts = float('nan')
            self.jw.add(np.concatenate([
                [now - self.t0, ts, snap['seq'], snap['fresh_age'], REC_MODE.get(app.mode, -1),
                 REC_SUB.get(sub, -1), float(ok)],
                snap['q'], snap['dq'], snap['quat'], snap['omega'],
                command[0], command[1], command[2], command[3], command[4], self.cmd, self.target,
                # --- v6 ---
                snap.get('tau', np.full(12, np.nan)), snap.get('acc', np.full(3, np.nan)),
                [snap.get('snap_us', float('nan'))]]))
        except Exception:
            self.errors += 1

    def policy(self, command, frame, raw, target, new_session=False):
        """After each 50 Hz inference: frame = the 45-d observation it used."""
        if self.closed:
            return
        try:
            if new_session:
                self.session += 1
                self.step = 0
            self.step += 1
            self.cmd, self.target = np.asarray(command, float).copy(), np.asarray(target, float).copy()
            self.pw.add(np.concatenate([[mono() - self.t0, self.step, self.session],
                                        self.cmd, frame, raw, self.target]))
        except Exception:
            self.errors += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        for w in (self.jw, self.pw):
            try:
                w.close()
            except Exception:
                self.errors += 1
        if self.errors:
            print('[rec] %d recorder errors (control not affected)' % self.errors, flush=True)


def start_walk_recorder(log_dir):
    """Create the recorder; on failure return None and run without recording."""
    try:
        return WalkRecorder(log_dir)
    except Exception as exc:
        print('[rec] could not start, running without recording: %r' % (exc,), flush=True)
        return None


class App:
    def __init__(self, args):
        self.a = args
        self.robot_ip = args.ip
        self.mode = 'connecting'
        self.m = None
        self.j = None
        self.pad = None
        self.gear = max(0, min(3, args.gear - 1))
        self.gait = 'walk'
        self.shutdown_requested = threading.Event()
        self.control_stop = threading.Event()
        self.state_lock = threading.Lock()
        self.snap_lock = threading.Lock()
        self.mode_lock = threading.RLock()
        self.sdk_lock = threading.RLock()  # serialize Joint SDK getters/send/disconnect
        self.motion_lock = threading.RLock()  # serialize Motion SDK reconnect/commands
        self.recovery_lock = threading.RLock()
        self.S = {
            'sub': 'damp', 'hs': None, 'goal': None, 't0': 0.0,
            'trajectory_generation': 0, 'prev_target': None,
            'release_t0': None, 'release_hold': None,
            'policy_generation': 0, 'policy_target': None,
            'policy_target_mono': 0.0, 'policy_last_action': np.zeros(12, np.float32),
            'stand_stable_since': None, 'policy_autostart_armed': False,
        }
        self.latest = None
        self.last_seq = None
        self.last_seq_arrival = 0.0
        self.policy_thread = None
        self.control_thread = None
        self.send_failures = 0
        self.motion_failures = 0
        self.joint_state_failures = 0
        self.fault_reason = None
        self.recovery_thread = None
        self._pad_resume_policy = False
        self._motion_velocity_enabled = False
        self._last_log_flush = mono()
        self._next_joint_udp_port = JOINT_UDP_PORT_MIN
        self.joint_udp_port = None
        self.rec = None  # walk recorder (created in start_control_thread)

    # ---------------- connection management ----------------
    def _mk_motion(self):
        client = bpx_sdk.MotionLevelControl()
        client.setRobotIp(self.robot_ip)
        client.setRobotStateUploadPort(9873)
        client.setTcpLocalPort(0)
        client.setRobotStateUploadRate(100)
        return client

    def _joint_udp_port_available(self, port):
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.bind(('0.0.0.0', int(port)))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    def _pick_joint_udp_port(self):
        count = JOINT_UDP_PORT_MAX - JOINT_UDP_PORT_MIN + 1
        for _ in range(count):
            port = self._next_joint_udp_port
            self._next_joint_udp_port += 1
            if self._next_joint_udp_port > JOINT_UDP_PORT_MAX:
                self._next_joint_udp_port = JOINT_UDP_PORT_MIN
            if self._joint_udp_port_available(port):
                return port
        raise RuntimeError('没有可用的 Joint 高频 UDP 端口（%d-%d）' %
                           (JOINT_UDP_PORT_MIN, JOINT_UDP_PORT_MAX))

    def _mk_joint(self):
        port = self._pick_joint_udp_port()
        client = bpx_sdk.JointLevelControl()
        try:
            client.setRobotIp(self.robot_ip)
            client.setRobotStateUploadPort(9873)
            client.setJointStateUploadPort(port)
            client.setTcpLocalPort(0)
            client.setRobotStateUploadRate(100)
        except Exception:
            self._dispose_joint_client(client)
            raise
        return client, port

    def _dispose_joint_client(self, client):
        """Best-effort teardown for both connected and failed SDK clients."""
        if client is None:
            return
        with self.sdk_lock:
            try:
                client.disconnect()
            except Exception:
                pass
        # Some bpx_sdk builds release the native receiver only from the wrapper
        # destructor. The caller clears its final Python reference and invokes GC.

    def connect_motion(self, attempts=None):
        self.mode = 'connecting'
        self.drop_joint()
        count = 0
        while not self.shutdown_requested.is_set():
            count += 1
            client = self._mk_motion()
            try:
                ok = bool(client.connect())
            except Exception as exc:
                ok = False
                if count == 1 or count % 10 == 0:
                    print('[WAIT] Motion connect 异常: %r' % (exc,), flush=True)
            if ok:
                with self.motion_lock:
                    self.m = client
                time.sleep(0.8)
                self.motion_failures = 0
                self.mode = 'motion'
                self.motion_safe_zero()
                print('[MOTION] 运控层已连接，motion=%s' % self.motion_state(), flush=True)
                return True
            try:
                client.disconnect()
            except Exception:
                pass
            if count == 1 or count % 10 == 0:
                print('[WAIT] 等待机器人 %s 上线（Motion，第 %d 次）…' % (self.robot_ip, count), flush=True)
            if attempts is not None and count >= attempts:
                return False
            self.shutdown_requested.wait(self.a.retry_interval)
        return False

    def _wait_for_high_rate(self, timeout=4.0):
        """Wait for two different high-rate sequence values.

        The robot may accept the Joint TCP session before its 1 kHz publisher
        starts sending to the newly selected UDP port.  The old code sampled
        only around one second and immediately destroyed the client, which made
        a slow robot look like a failed DevelopingState transition.
        """
        first = None
        deadline = mono() + max(0.5, float(timeout))
        while not self.shutdown_requested.is_set() and mono() < deadline:
            snap = self.read_high_snapshot(retries=3)
            if snap is not None:
                if first is None:
                    first = snap
                elif snap['seq'] != first['seq']:
                    return first, snap
            time.sleep(0.05)
        return first, None

    def connect_joint(self, attempts=40):
        self.motion_safe_zero()
        self.drop_motion()
        self.mode = 'transition'
        count = 0
        while attempts is None or count < attempts:
            count += 1
            if self.shutdown_requested.is_set():
                return False
            try:
                client, joint_port = self._mk_joint()
                print('[JOINT] 使用高频状态 UDP 端口 %d' % joint_port, flush=True)
            except Exception as exc:
                print('[门1] Joint 客户端创建失败: %r' % (exc,), flush=True)
                time.sleep(self.a.retry_interval)
                continue
            try:
                ok = bool(client.connect())
            except Exception as exc:
                ok = False
                print('[门1] Joint connect 异常: %r' % (exc,), flush=True)
            if ok:
                self.j = client
                self.joint_udp_port = joint_port
                first, second = self._wait_for_high_rate(timeout=4.0)
                if first is not None and second is not None:
                    with self.state_lock:
                        self.S.update(sub='damp', hs=None, goal=None, prev_target=None,
                                      release_t0=None, release_hold=None,
                                      stand_stable_since=None, policy_autostart_armed=False)
                    self.joint_state_failures = 0
                    self.send_failures = 0
                    self.mode = 'joint'
                    print('[JOINT] DevelopingState 已确认（高频序列 %s→%s，motion=%s）' %
                          (first['seq'], second['seq'], self.motion_state()), flush=True)
                    return True
                print('[门1] 已连接但高频状态未推进，拆除后重试', flush=True)
                self.drop_joint()
            else:
                self._dispose_joint_client(client)
                client = None
                gc.collect()
            self.shutdown_requested.wait(self.a.retry_interval)
        return False

    def request_robot_reconnect(self, reason):
        """Reconnect the active SDK session without terminating the process.

        Recovery always enters a safe state first.  A previous POLICY session is
        restored through the normal stand-up/0.5 s stability gate after JOINT
        telemetry is live again; this avoids sending stale targets across a
        broken Ethernet session.
        """
        if self.shutdown_requested.is_set():
            return
        with self.recovery_lock:
            if self.recovery_thread is not None and self.recovery_thread.is_alive():
                return
            if self.mode not in ('motion', 'joint', 'policy'):
                return
            previous_mode = self.mode
            with self.state_lock:
                previous_sub = self.S['sub']
            thread = threading.Thread(
                target=self._robot_reconnect_worker,
                args=(reason, previous_mode, previous_sub),
                name='bpx-robot-reconnect', daemon=True)
            self.recovery_thread = thread
            thread.start()

    def _robot_reconnect_worker(self, reason, previous_mode, previous_sub):
        resume_policy = previous_mode == 'policy' or previous_sub == 'policy'
        print('[NET] %s；进入安全状态并等待机器人网络恢复…' % reason, flush=True)
        try:
            if previous_mode == 'motion':
                self.mode = 'reconnecting'
                self.motion_safe_zero()
                self.drop_motion()
                if self.connect_motion(attempts=None):
                    print('[NET] MOTION 已重连；保持原档位/步态，等待手柄继续控制', flush=True)
                return

            # JOINT/POLICY: stop policy first, then rebuild the high-rate UDP
            # receiver with a fresh local port.  Never reuse the stale client.
            self.stop_policy('网络重连')
            self.request_damp('网络重连')
            self.drop_joint()
            if not self.connect_joint(attempts=None):
                return
            if resume_policy:
                # Re-enter through the existing PD/Y safety path instead of
                # replaying a stale action/history buffer.
                if self.request_pd(STAND_POSE, '网络重连恢复', auto_start_policy=True):
                    print('[NET] JOINT 已重连；PD站稳后自动恢复 POLICY', flush=True)
                else:
                    print('[NET] JOINT 已重连，但未自动恢复 POLICY；请按 Y', flush=True)
            else:
                print('[NET] JOINT 已重连；当前为安全阻尼，请按 Y 继续', flush=True)
        except Exception as exc:
            print('[NET] 重连线程异常: %r；继续等待下一次故障触发' % (exc,), flush=True)
        finally:
            with self.recovery_lock:
                self.recovery_thread = None

    def drop_motion(self):
        with self.motion_lock:
            client = self.m
            self.m = None
            if client is None:
                return
            try:
                client.setVelocity(0.0, 0.0, 0.0)
            except Exception:
                pass
            try:
                client.setVelocityControlFlag(False)
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass
            self._motion_velocity_enabled = False

    def drop_joint(self):
        with self.sdk_lock:
            client = self.j
            self.j = None
            self.joint_udp_port = None
        self._dispose_joint_client(client)
        client = None
        gc.collect()
        with self.snap_lock:
            self.latest = None
            self.last_seq = None
            self.last_seq_arrival = 0.0

    # ---------------- input/state ----------------
    def motion_state(self):
        if self.j is not None:
            try:
                return self.j.getCurrentMotionState()
            except Exception:
                pass
        with self.motion_lock:
            client = self.m
            if client is None:
                return None
            try:
                return client.getCurrentMotionState()
            except Exception:
                return None

    def read_motion_q(self):
        with self.motion_lock:
            client = self.m
            if client is None:
                return None
            try:
                return vec(client.getJointPosition(), 12, 'motion_joint_position')
            except Exception:
                return None

    def read_high_snapshot(self, retries=3):
        for _ in range(max(1, retries)):
            try:
                with self.sdk_lock:
                    client = self.j
                    if client is None:
                        return None
                    seq0 = int(client.getJointStateSeqHighRate())
                    _snap_t0 = mono()
                    q = vec(client.getJointPositionHighRate(), 12, 'joint_position_high_rate')
                    dq = vec(client.getJointVelocityHighRate(), 12, 'joint_velocity_high_rate')
                    omega = vec(client.getImuOmegaHighRate(), 3, 'imu_omega_high_rate')
                    quat = vec(client.getImuQuatHighRate(), 4, 'imu_quat_high_rate')
                    # --- v6 新增：力估计器要的两组量 ---
                    # 读不到就记 NaN，**绝不让采数据的附加项拖垮控制回路**。
                    # 不包这层 try 的话：getter 返回 None -> vec() 抛异常 -> 被外面的
                    # except Exception: continue 吞掉 -> 重试三次后整个快照返回 None
                    # -> 控制循环报「状态未就绪/超时」。发生在走路中间就是控制断链。
                    tau, acc = np.full(12, np.nan), np.full(3, np.nan)
                    if RECORD_EXTRA_SENSORS:
                        try:
                            tau = vec(client.getJointTorqueHighRate(), 12, 'joint_torque_high_rate')
                            acc = vec(client.getImuAccHighRate(), 3, 'imu_acc_high_rate')
                        except Exception:
                            pass
                    robot_ts = client.getJointStateTimestampHighRate()
                    seq1 = int(client.getJointStateSeqHighRate())
                    _snap_us = (mono() - _snap_t0) * 1e6
                if seq0 != seq1:
                    continue
                now = mono()
                snap = {
                    'seq': seq1, 'q': q, 'dq': dq, 'omega': omega, 'quat': quat,
                    'robot_ts': robot_ts, 'arrival': now,
                    'tau': tau, 'acc': acc, 'snap_us': _snap_us,   # v6
                }
                with self.snap_lock:
                    if self.last_seq != seq1:
                        self.last_seq = seq1
                        self.last_seq_arrival = now
                    snap['fresh_age'] = now - self.last_seq_arrival
                    self.latest = snap
                return snap
            except Exception:
                continue
        return None

    def snapshot(self):
        with self.snap_lock:
            if self.latest is None:
                return None
            out = dict(self.latest)
            out['fresh_age'] = mono() - self.last_seq_arrival if self.last_seq_arrival else float('inf')
            return out

    def current_q(self):
        snap = self.snapshot()
        if snap is not None and snap['fresh_age'] <= STATE_TIMEOUT:
            return snap['q'].copy()
        return self.read_motion_q()

    def is_lying(self):
        q = self.current_q()
        if q is None:
            return False
        near = float(np.max(np.abs(q - LIE_POSE))) < 0.5
        return near and self.motion_state() in (0, 2, 7, None)

    # ---------------- joint command generation ----------------
    @staticmethod
    def checked_command(kp, target, kd, velocity, tff):
        arrays = [
            vec(kp, 12, 'kp'), vec(target, 12, 'target'), vec(kd, 12, 'kd'),
            vec(velocity, 12, 'velocity'), vec(tff, 12, 'tff'),
        ]
        if np.any(arrays[0] < 0.0) or np.any(arrays[2] < 0.0):
            raise ValueError('kp/kd 不能为负')
        if np.max(arrays[0]) > 60.0 or np.max(arrays[2]) > 10.0:
            raise ValueError('kp/kd 超出部署上限')
        return arrays

    @staticmethod
    def torque_limited_pd(target, q, dq, kp, kd, tff):
        kp_a = np.full(12, kp, np.float32) if np.isscalar(kp) else vec(kp, 12, 'kp')
        kd_a = np.full(12, kd, np.float32) if np.isscalar(kd) else vec(kd, 12, 'kd')
        ff_a = vec(tff, 12, 'tff')
        desired_tau = kp_a * (target - q) - kd_a * dq + ff_a
        limited_tau = np.clip(desired_tau, -EFFORT_LIMIT, EFFORT_LIMIT)
        q_cmd = target.copy()
        active = kp_a > 1e-6
        q_cmd[active] = q[active] + (limited_tau[active] + kd_a[active] * dq[active] - ff_a[active]) / kp_a[active]
        return kp_a, q_cmd.astype(np.float32), kd_a, np.zeros(12, np.float32), ff_a

    def request_pd(self, goal, label, auto_start_policy=False):
        snap = self.snapshot()
        if snap is None or snap['fresh_age'] > STATE_TIMEOUT:
            print('[JOINT] %s 被拒绝：高频状态未就绪/已超时' % label, flush=True)
            return False
        q = snap['q']
        dp_lie = float(np.max(np.abs(q - LIE_POSE)))
        dp_stand = float(np.max(np.abs(q - STAND_POSE)))
        if float(np.max(np.abs(q))) < 0.15 or min(dp_lie, dp_stand) > 0.6:
            print('[JOINT] %s 被拒绝：姿态异常（离趴姿 %.2f，离站姿 %.2f）' %
                  (label, dp_lie, dp_stand), flush=True)
            return False
        self.stop_policy('切换到 PD')
        with self.state_lock:
            self.S['trajectory_generation'] += 1
            self.S.update(sub='pd', hs=q.copy(), goal=goal.copy(), t0=mono(),
                          prev_target=q.copy(), release_t0=None, release_hold=None,
                          stand_stable_since=None,
                          policy_autostart_armed=bool(auto_start_policy))
        self.mode = 'joint'
        return True

    def request_damp(self, reason):
        self.stop_policy(reason)
        with self.state_lock:
            self.S.update(sub='damp', hs=None, goal=None, prev_target=None,
                          release_t0=None, release_hold=None, stand_stable_since=None,
                          policy_autostart_armed=False)
        if self.j is not None:
            self.mode = 'joint'
        print('[JOINT] 阻尼安全状态：%s' % reason, flush=True)

    def request_release(self):
        self.stop_policy('退出/渐进放松')
        snap = self.snapshot()
        hold = None if snap is None else snap['q'].copy()
        with self.state_lock:
            self.S.update(sub='release', release_t0=mono(), release_hold=hold,
                          stand_stable_since=None, policy_autostart_armed=False)
        if self.j is not None:
            self.mode = 'joint'

    def command_from_state(self, snap, now):
        q, dq = snap['q'], snap['dq']
        with self.state_lock:
            sub = self.S['sub']
            hs = None if self.S['hs'] is None else self.S['hs'].copy()
            goal = None if self.S['goal'] is None else self.S['goal'].copy()
            t0 = self.S['t0']
            prev = None if self.S['prev_target'] is None else self.S['prev_target'].copy()
            release_t0 = self.S['release_t0']
            release_hold = None if self.S['release_hold'] is None else self.S['release_hold'].copy()
            policy_target = None if self.S['policy_target'] is None else self.S['policy_target'].copy()
            policy_age = now - self.S['policy_target_mono']

        if sub == 'pd' and hs is not None and goal is not None:
            elapsed = now - t0
            progress = smoothstep(elapsed / TS_PD)
            target = hs * (1.0 - progress) + goal * progress
            if prev is not None:
                target = prev + np.clip(target - prev, -SLEW / HZ_JOINT, SLEW / HZ_JOINT)
            with self.state_lock:
                self.S['prev_target'] = target.copy()
            sitting = float(np.max(np.abs(goal - LIE_POSE))) < 0.05
            tff = gravity_tff(target) * ((1.0 - progress) if sitting else 1.0)
            kp, kd = KP_PD, KD_PD
            if sitting and elapsed >= TS_PD + self.a.passive_delay:
                release_progress = min(1.0, (elapsed - TS_PD - self.a.passive_delay) / max(0.05, RELEASE))
                weight = 1.0 - smoothstep(release_progress)
                kp = KP_PD * weight
                kd = KD_PD * weight + KD_DAMP * (1.0 - weight)
                tff *= weight
                if release_progress >= 1.0:
                    with self.state_lock:
                        self.S.update(sub='damp', hs=None, goal=None, prev_target=None)
                    print('[JOINT] 趴下渐进放松完成 → 阻尼', flush=True)
            if not sitting and elapsed >= TS_PD:
                err = float(np.max(np.abs(q - STAND_POSE)))
                speed = float(np.max(np.abs(dq)))
                try:
                    gravity = projected_gravity_xyzw(snap['quat'])
                    tilt_ok = gravity[2] < -0.75
                except Exception:
                    tilt_ok = False
                stable = err < 0.25 and speed < 0.5 and tilt_ok
                with self.state_lock:
                    if stable and self.S['stand_stable_since'] is None:
                        self.S['stand_stable_since'] = now
                    elif not stable:
                        self.S['stand_stable_since'] = None
            return self.torque_limited_pd(target, q, dq, kp, kd, tff)

        if sub == 'policy':
            if policy_target is None or policy_age > POLICY_TARGET_TIMEOUT:
                raise RuntimeError('策略目标超时 %.3fs' % policy_age)
            zeros = np.zeros(12, np.float32)
            return self.torque_limited_pd(policy_target, q, dq, KP_POLICY, KD_POLICY, zeros)

        if sub == 'release':
            if release_t0 is None:
                release_t0 = now
                with self.state_lock:
                    self.S['release_t0'] = now
            target = q.copy() if release_hold is None else release_hold
            progress = min(1.0, (now - release_t0) / max(0.05, RELEASE))
            weight = 1.0 - smoothstep(progress)
            kp = np.full(12, KP_PD * weight, np.float32)
            kd = np.full(12, KD_PD * weight + KD_DAMP * (1.0 - weight), np.float32)
            cmd = self.torque_limited_pd(target, q, dq, kp, kd, np.zeros(12, np.float32))
            if progress >= 1.0:
                with self.state_lock:
                    self.S.update(sub='damp', release_t0=None, release_hold=None)
            return cmd

        # damp / fault: hold current measured position with kp=0, kd=3.
        return (np.zeros(12, np.float32), q.copy(), np.full(12, KD_DAMP, np.float32),
                np.zeros(12, np.float32), np.zeros(12, np.float32))

    def control_loop(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / 'dual_pad_traj.csv'
        dt = 1.0 / HZ_JOINT
        deadline = mono()
        deadline_misses = 0
        max_late = 0.0
        row_count = 0
        with log_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['mono', 'mode', 'sub', 'seq', 'state_age', 'send_ok', 'deadline_misses', 'max_late_ms'] +
                            ['q%d' % i for i in range(12)] + ['dq%d' % i for i in range(12)] +
                            ['target%d' % i for i in range(12)])
            while not self.control_stop.is_set():
                deadline += dt
                if self.j is None or self.mode not in ('joint', 'policy'):
                    time.sleep(0.002)
                    deadline = mono()
                    continue
                snap = self.read_high_snapshot(retries=4)
                if snap is None:
                    snap = self.snapshot()
                now = mono()
                if snap is None or snap['fresh_age'] > STATE_TIMEOUT:
                    self.joint_state_failures += 1
                    if self.joint_state_failures >= JOINT_RECONNECT_FAILURES:
                        self.request_robot_reconnect('JOINT 高频状态连续超时')
                    if self.fault_reason != '高频状态超时':
                        self.fault_reason = '高频状态超时'
                        print('[WATCHDOG] 高频状态超时；停止策略并重复发送阻尼帧', flush=True)
                    with self.state_lock:
                        self.S.update(sub='damp', policy_target=None, stand_stable_since=None,
                                      policy_autostart_armed=False)
                    self.mode = 'joint'
                    # Do not leave the last policy/PD frame latched merely because
                    # telemetry stopped.  kp=0 makes the stale position irrelevant.
                    if snap is not None:
                        damp = self.checked_command(
                            np.zeros(12), snap['q'], np.full(12, KD_DAMP),
                            np.zeros(12), np.zeros(12))
                        try:
                            with self.sdk_lock:
                                client = self.j
                                if client is not None:
                                    client.setJointCommand(
                                        *[[float(v) for v in arr] for arr in damp])
                        except Exception:
                            pass
                    time.sleep(0.002)
                    deadline = mono()
                    continue
                self.joint_state_failures = 0
                try:
                    command = self.checked_command(*self.command_from_state(snap, now))
                except Exception as exc:
                    print('[WATCHDOG] 命令生成失败: %r → 阻尼' % (exc,), flush=True)
                    with self.state_lock:
                        self.S.update(sub='damp', policy_target=None, stand_stable_since=None,
                                      policy_autostart_armed=False)
                    self.mode = 'joint'
                    command = self.checked_command(
                        np.zeros(12), snap['q'], np.full(12, KD_DAMP), np.zeros(12), np.zeros(12))
                try:
                    with self.sdk_lock:
                        client = self.j
                        ok = client is not None and bool(client.setJointCommand(
                            *[[float(v) for v in arr] for arr in command]))
                except Exception as exc:
                    ok = False
                    if self.send_failures == 0:
                        print('[WATCHDOG] setJointCommand 异常: %r' % (exc,), flush=True)
                self.send_failures = 0 if ok else self.send_failures + 1
                if self.send_failures >= MAX_SEND_FAILURES:
                    if self.fault_reason != '连续发送失败':
                        print('[WATCHDOG] 连续发送失败 → 阻尼/等待人工处理', flush=True)
                    self.fault_reason = '连续发送失败'
                    with self.state_lock:
                        self.S.update(sub='damp', policy_target=None, stand_stable_since=None,
                                      policy_autostart_armed=False)
                    self.mode = 'joint'
                    self.request_robot_reconnect('JOINT 命令连续发送失败')
                if self.rec is not None:
                    self.rec.joint(self, now, snap, command, ok)

                row_count += 1
                if row_count % 5 == 0:  # 100 Hz audit logging, no per-cycle flush.
                    with self.state_lock:
                        sub = self.S['sub']
                    writer.writerow([
                        '%.6f' % now, self.mode, sub, snap['seq'], '%.6f' % snap['fresh_age'],
                        int(ok), deadline_misses, '%.3f' % (max_late * 1000.0),
                    ] + ['%.6f' % x for x in snap['q']] + ['%.6f' % x for x in snap['dq']] +
                       ['%.6f' % x for x in command[1]])
                if now - self._last_log_flush >= 1.0:
                    handle.flush()
                    self._last_log_flush = now

                sleep_for = deadline - mono()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    late = -sleep_for
                    deadline_misses += 1
                    max_late = max(max_late, late)
                    deadline = mono()

    # ---------------- policy ----------------
    def convention_confirmed(self):
        try:
            import yaml
            data = yaml.safe_load(POLICY_CFG.read_text())
            return data.get('hardware_joint_convention_confirmed') is True
        except Exception as exc:
            print('[门2] 无法读取安全闸门: %r' % (exc,), flush=True)
            return False

    def build_obs_frame(self, command, last_action, snap=None):
        if snap is None:
            snap = self.snapshot()
        if snap is None or snap['fresh_age'] > STATE_TIMEOUT:
            raise RuntimeError('状态未就绪/超时')
        gravity = projected_gravity_xyzw(snap['quat'])
        frame = np.concatenate([
            snap['q'] - DEFAULT_POSE,
            snap['dq'],
            snap['omega'],
            gravity,
            vec(last_action, 12, 'last_action'),
            vec(command, 3, 'command'),
        ]).astype(np.float32)
        return vec(frame, POLICY_FRAME_SIZE, 'policy_frame')

    def make_policy_session(self):
        import onnxruntime as ort
        session = ort.InferenceSession(str(POLICY_ONNX), providers=['CPUExecutionProvider'])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 1:
            raise ValueError('RMA ONNX 必须有 2 个输入和 1 个输出')
        inputs_by_name = {item.name: item for item in inputs}
        if set(inputs_by_name) != {'policy', 'adaptation_history'}:
            raise ValueError('ONNX 输入名称不符: %s' % sorted(inputs_by_name))
        out = outputs[0]
        if out.name != 'actions':
            raise ValueError('ONNX 输出名称不符: %s' % out.name)
        if any(item.type != 'tensor(float)' for item in inputs) or out.type != 'tensor(float)':
            raise ValueError('ONNX 输入输出必须为 float32')
        shapes = {name: list(item.shape) for name, item in inputs_by_name.items()}
        expected = {'policy': [1, POLICY_FRAME_SIZE],
                    'adaptation_history': [1, POLICY_HISTORY_INPUT_SIZE]}
        if shapes != expected or list(out.shape) != [1, 12]:
            raise ValueError('ONNX shape 不符: inputs=%s output=%s' % (shapes, out.shape))
        return session, tuple(expected)

    @staticmethod
    def flatten_term_major_history(history):
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (POLICY_HISTORY_LENGTH, POLICY_FRAME_SIZE):
            raise ValueError('策略历史 shape=%s，期望 (%d,%d)' %
                             (history.shape, POLICY_HISTORY_LENGTH, POLICY_FRAME_SIZE))
        flat = np.concatenate([
            history[:, begin:end].reshape(-1)
            for begin, end in POLICY_TERM_SLICES
        ]).astype(np.float32)
        return vec(flat, POLICY_HISTORY_INPUT_SIZE, 'adaptation_history')

    @staticmethod
    def run_policy(session, input_names, policy_frame, history):
        if set(input_names) != {'policy', 'adaptation_history'}:
            raise ValueError('RMA 输入名称不符: %s' % (input_names,))
        policy = vec(policy_frame, POLICY_FRAME_SIZE, 'policy').reshape(1, -1)
        adaptation_history = App.flatten_term_major_history(history).reshape(1, -1)
        network_action = vec(session.run(None, {
            'policy': policy,
            'adaptation_history': adaptation_history,
        })[0], 12, 'policy_action')
        # The exported graph already contains this clamp. Repeat it here as a
        # fail-safe so last_action, target generation, and training use exactly
        # the same bounded raw-action contract.
        raw = np.clip(network_action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)
        target = np.clip(DEFAULT_POSE + ACTION_SCALE * raw,
                         MODEL_LIMITS[:, 0], MODEL_LIMITS[:, 1]).astype(np.float32)
        return raw, target

    def prepare_policy_state(self, session, input_names, command):
        # Match mjlab reset: backfill all 25 history slots with the first frame.
        state = {'last_action': np.zeros(12, dtype=np.float32)}
        frame = self.build_obs_frame(command, state['last_action'])
        state['history'] = np.repeat(frame.reshape(1, -1), POLICY_HISTORY_LENGTH, axis=0)
        raw, target = self.run_policy(session, input_names, frame, state['history'])
        state['last_action'] = raw
        return state, raw, target

    def step_policy_state(self, session, input_names, state, command):
        frame = self.build_obs_frame(command, state['last_action'])
        state['history'][:-1] = state['history'][1:]
        state['history'][-1] = frame
        raw, target = self.run_policy(session, input_names, frame, state['history'])
        state['last_action'] = raw
        return raw, target

    def policy_ready(self):
        snap = self.snapshot()
        if snap is None or snap['fresh_age'] > STATE_TIMEOUT:
            return False, '高频状态未就绪/超时'
        with self.state_lock:
            sub = self.S['sub']
            stable_since = self.S['stand_stable_since']
        err = float(np.max(np.abs(snap['q'] - STAND_POSE)))
        speed = float(np.max(np.abs(snap['dq'])))
        if sub != 'pd':
            return False, '需要先用 Y 完成关节层站立'
        if stable_since is None or mono() - stable_since < 0.5:
            return False, '站立尚未连续稳定 0.5s（误差 %.3f，速度 %.3f）' % (err, speed)
        return True, ''

    def maybe_autostart_policy(self):
        # ONNX session creation can block, so this runs from the 100 Hz main
        # loop rather than the 500 Hz joint command thread.
        with self.state_lock:
            armed = self.S['policy_autostart_armed']
        if self.mode != 'joint' or not armed:
            return False
        ready, _reason = self.policy_ready()
        if not ready:
            return False
        # Consume the one-shot before loading ONNX. A failed preparation must
        # not retry every 10 ms; the operator can press Y again or use RB.
        with self.state_lock:
            if self.mode != 'joint' or not self.S['policy_autostart_armed']:
                return False
            self.S['policy_autostart_armed'] = False
        return self.start_policy('Y站稳自动')

    def start_policy(self, trigger='RB'):
        if self.mode == 'policy':
            print('[POLICY] 已在策略模式', flush=True)
            return False
        ready, reason = self.policy_ready()
        if not ready:
            print('[门2] %s 被拒绝：%s' % (trigger, reason), flush=True)
            return False
        if not self.convention_confirmed():
            print('[门2] %s 被拒绝：hardware_joint_convention_confirmed=false' % trigger, flush=True)
            return False
        if not POLICY_ONNX.is_file():
            print('[门2] 策略不存在: %s' % POLICY_ONNX, flush=True)
            return False
        try:
            session, input_names = self.make_policy_session()
            command = self.policy_cmd()
            policy_state, raw, target = self.prepare_policy_state(session, input_names, command)
        except Exception as exc:
            print('[门2] 策略准备/首帧推理失败: %r' % (exc,), flush=True)
            return False
        with self.state_lock:
            self.S['policy_generation'] += 1
            generation = self.S['policy_generation']
            self.S.update(sub='policy', policy_target=target, policy_target_mono=mono(),
                          policy_last_action=raw, stand_stable_since=None,
                          policy_autostart_armed=False)
        if self.rec is not None:
            self.rec.policy(command, policy_state['history'][-1], raw, target, new_session=True)
        self.mode = 'policy'  # atomic handoff: never enters damping first.
        self.policy_thread = threading.Thread(
            target=self.policy_loop, args=(session, input_names, generation, policy_state),
            name='bpx-policy-50hz', daemon=True)
        self.policy_thread.start()
        print('[门2] %s → POLICY（档位 %d，前进满杆 %.1f m/s；50Hz 推理/500Hz 下发）' %
              (trigger, self.gear + 1, POLICY_VX_FWD_MAX * GEARS[self.gear]), flush=True)
        return True

    def stop_policy(self, reason):
        with self.state_lock:
            was_policy = self.S['sub'] == 'policy' or self.mode == 'policy'
            self.S['policy_generation'] += 1
            self.S['policy_target'] = None
            self.S['policy_target_mono'] = 0.0
            self.S['policy_autostart_armed'] = False
        if was_policy:
            self.mode = 'joint'
            print('[POLICY] 停止：%s' % reason, flush=True)
        thread = self.policy_thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=0.2)
        self.policy_thread = None

    def policy_loop(self, session, input_names, generation, policy_state):
        dt = 1.0 / HZ_POLICY
        deadline = mono() + dt
        try:
            while not self.shutdown_requested.is_set():
                wait = deadline - mono()
                if wait > 0:
                    time.sleep(wait)
                with self.state_lock:
                    if generation != self.S['policy_generation'] or self.S['sub'] != 'policy':
                        return
                command = self.policy_cmd()
                raw, target = self.step_policy_state(
                    session, input_names, policy_state, command)
                with self.state_lock:
                    if generation != self.S['policy_generation'] or self.S['sub'] != 'policy':
                        return
                    self.S.update(policy_target=target, policy_target_mono=mono(), policy_last_action=raw)
                if self.rec is not None:
                    self.rec.policy(command, policy_state['history'][-1], raw, target)
                deadline += dt
                if deadline < mono():
                    deadline = mono() + dt
        except Exception as exc:
            print('[POLICY-WATCHDOG] 推理/观测异常: %r' % (exc,), flush=True)
            with self.state_lock:
                if generation == self.S['policy_generation']:
                    self.S['policy_generation'] += 1
                    self.S.update(sub='damp', policy_target=None, stand_stable_since=None,
                                      policy_autostart_armed=False)
                    self.mode = 'joint'

    # ---------------- operator commands ----------------
    def policy_cmd(self):
        if self.pad is None or not self.pad.online:
            return np.zeros(3, np.float32)
        ly = -self.pad.norm(0x01, -32767, 32767)
        lx = self.pad.norm(0x00, -32767, 32767)
        rx = self.pad.norm(0x03, -32767, 32767)
        factor = GEARS[self.gear]
        vx_limit = POLICY_VX_FWD_MAX if ly >= 0.0 else POLICY_VX_BACK_MAX
        return np.array([
            ly * vx_limit * factor,
            -lx * POLICY_VY_MAX * factor,
            -rx * POLICY_YAW_MAX * factor,
        ], dtype=np.float32)

    def motion_velocity(self):
        if self.pad is None or not self.pad.online:
            return 0.0, 0.0, 0.0
        ly = -self.pad.norm(0x01, -32767, 32767)
        lx = self.pad.norm(0x00, -32767, 32767)
        rx = self.pad.norm(0x03, -32767, 32767)
        vx_max, vy_max, yaw_max = MOTION_GAIT_LIM[self.gait]
        factor = GEARS[self.gear]
        # Linux ABS_X/ABS_RX increase to the physical right, while the BPX
        # velocity API uses positive lateral/yaw toward the physical left.
        # Match the POLICY convention so left/right remains identical in both layers.
        return (
            ly * min(vx_max, MOTION_CAPS['vx']) * factor,
            -lx * min(vy_max, MOTION_CAPS['vy']) * factor,
            -rx * min(yaw_max, MOTION_CAPS['yaw']) * factor,
        )

    def motion_safe_zero(self):
        with self.motion_lock:
            client = self.m
            if client is None:
                self._motion_velocity_enabled = False
                return
            try:
                client.setVelocity(0.0, 0.0, 0.0)
            except Exception:
                pass
            try:
                client.setVelocityControlFlag(False)
            except Exception:
                pass
            self._motion_velocity_enabled = False

    def motion_tick(self):
        if self.mode != 'motion' or self.m is None:
            return
        vx, vy, yaw = self.motion_velocity()
        state = self.motion_state()
        if state is None:
            self.motion_failures += 1
            if self.motion_failures >= MOTION_RECONNECT_FAILURES:
                self.request_robot_reconnect('MOTION 状态读取失败')
            return
        self.motion_failures = 0
        if state == 6:
            try:
                with self.motion_lock:
                    client = self.m
                    if client is None:
                        return
                    client.setVelocityControlFlag(True)
                    self._motion_velocity_enabled = True
                    client.setVelocity(float(vx), float(vy), float(yaw))  # explicit zero included
                    if self.gait == 'running':
                        client.setRunning()
                    else:
                        client.setWalk()
                self.motion_failures = 0
            except Exception as exc:
                self.motion_failures += 1
                print('[MOTION-WATCHDOG] 速度发送失败: %r' % (exc,), flush=True)
                self.motion_safe_zero()
                if self.motion_failures >= MOTION_RECONNECT_FAILURES:
                    self.request_robot_reconnect('MOTION 速度发送连续失败')
        elif self._motion_velocity_enabled:
            self.motion_safe_zero()

    def update_gears(self):
        if self.pad is None:
            return
        lt = self.pad.axes.get(0x02, 0) / 255.0
        rt = self.pad.axes.get(0x05, 0) / 255.0
        old_lt, old_rt = getattr(self, '_lt', False), getattr(self, '_rt', False)
        if lt > 0.5 and not old_lt:
            self.gear = max(0, self.gear - 1)
            print('[档位] %d/4 (x%.2f)' % (self.gear + 1, GEARS[self.gear]), flush=True)
        if rt > 0.5 and not old_rt:
            self.gear = min(3, self.gear + 1)
            print('[档位] %d/4 (x%.2f)' % (self.gear + 1, GEARS[self.gear]), flush=True)
        self._lt, self._rt = lt > 0.5, rt > 0.5

    def on_pad_fault(self):
        # A lost pad is recoverable.  Keep the process and SDK session alive,
        # but remove all operator velocity immediately.  JOINT/POLICY is put in
        # damping; a previous POLICY session is rebuilt through the normal PD
        # stand gate after the new device appears.
        self._pad_resume_policy = self.mode == 'policy'
        if self.mode == 'motion':
            self.motion_safe_zero()
            print('[PAD] 已进入零速度安全状态，等待手柄重新连接…', flush=True)
        elif self.mode in ('joint', 'policy'):
            self.request_damp('手柄失联')
        else:
            print('[PAD] 当前模式=%s，等待手柄重新连接…' % self.mode, flush=True)

    def on_pad_reconnected(self):
        self._lt = self._rt = False
        print('[PAD] 手柄已重连；速度/按键状态已清零', flush=True)
        if self._pad_resume_policy and self.mode == 'joint' and self.j is not None:
            self._pad_resume_policy = False
            if self.request_pd(STAND_POSE, '手柄重连恢复', auto_start_policy=True):
                print('[PAD] PD站稳后自动恢复 POLICY', flush=True)
            else:
                print('[PAD] 未自动恢复 POLICY；请按 Y 重新站立', flush=True)
        else:
            self._pad_resume_policy = False

    def handle_button(self, button):
        name = BTN_NAME.get(button, hex(button))
        if button == 0x13A:
            print_help()
            return
        if button == 0x13B:
            print('[PAD] START → 安全退出', flush=True)
            self.shutdown_requested.set()
            return

        if self.mode == 'motion':
            if button == 0x134:
                print('[MOTION] Y → setStandUp()=%s' % self.m.setStandUp(), flush=True)
            elif button == 0x130:
                self.motion_safe_zero()
                print('[MOTION] A → setSitDown()=%s' % self.m.setSitDown(), flush=True)
            elif button in (0x133, 0x13E):
                self.motion_safe_zero()
                print('[MOTION] %s → setDamping()=%s' % (name, self.m.setDamping()), flush=True)
            elif button == 0x136:
                self.gait = 'walk'
                print('[MOTION] LB → 普通行走，result=%s' % self.m.setWalk(), flush=True)
            elif button == 0x137:
                if self.is_lying():
                    print('[门1] 趴姿确认 → MOTION 切 JOINT…', flush=True)
                    if self.connect_joint():
                        print('[JOINT] 就绪：Y站立并自动进策略/A趴下/RB手动补启/LB回MOTION', flush=True)
                    else:
                        print('[门1] JOINT 连接失败，尝试恢复 MOTION', flush=True)
                        if not self.connect_motion(attempts=40):
                            self.mode = 'fault'
                            self.fault_reason = 'MOTION/JOINT 均连接失败'
                            self.shutdown_requested.set()
                else:
                    self.gait = 'running'
                    print('[MOTION] RB → 跑步，result=%s' % self.m.setRunning(), flush=True)
            return

        if self.mode not in ('joint', 'policy'):
            print('[TRANSITION] 当前状态 %s，忽略按键 %s' % (self.mode, name), flush=True)
            return
        if button == 0x134:
            # Y is also a recovery command while POLICY is active: invalidate the
            # policy generation first, then hand the same joint-PD stand
            # trajectory to the 500 Hz thread.  Previously Y was unconditionally
            # rejected in POLICY, leaving the operator with A as the only way out
            # even when the robot had stopped or failed to stand.
            if self.mode == 'policy':
                print('[JOINT] Y → 停止策略并重新执行 PD 站立', flush=True)
            if self.request_pd(STAND_POSE, 'Y', auto_start_policy=True):
                print('[JOINT] Y → PD 站立（3s），站稳 0.5s 后自动进入 POLICY', flush=True)
        elif button == 0x130:
            if self.request_pd(LIE_POSE, 'A'):
                print('[JOINT] A → PD 趴下，随后渐进放松', flush=True)
        elif button == 0x137:
            self.start_policy()
        elif button == 0x136:
            with self.state_lock:
                sub = self.S['sub']
            if self.mode == 'policy':
                print('[门3] LB 被拒绝：策略运行中，先按 A', flush=True)
            elif not (sub == 'damp' and self.is_lying()):
                print('[门3] LB 被拒绝：需先 A 趴下并完成渐进放松（当前 %s）' % sub, flush=True)
            else:
                print('[门3] 趴姿+阻尼确认 → JOINT 切 MOTION…', flush=True)
                self.stop_policy('切回 MOTION')
                self.mode = 'transition'
                self.drop_joint()
                # A cable swap or a slow SDK disconnect can make the first
                # Motion reconnect attempts fail.  Keep the controller in a
                # safe transition and wait for the robot instead of turning a
                # transient reconnect failure into a terminal fault/exit.
                if self.connect_motion(attempts=None):
                    print('[MOTION] 已恢复：Y站/A趴/LB走/RB跑', flush=True)
                else:
                    self.mode = 'fault'
                    self.fault_reason = '恢复 MOTION 失败'
                    self.shutdown_requested.set()
        elif button in (0x133, 0x13E):
            self.request_damp('%s 急停' % name)

    # ---------------- lifecycle ----------------
    def start_control_thread(self):
        if not getattr(self.a, 'no_record', False):
            self.rec = start_walk_recorder(LOG_DIR)
        self.control_thread = threading.Thread(target=self.control_loop, name='bpx-joint-500hz', daemon=True)
        self.control_thread.start()

    def shutdown(self):
        print('[退出] 开始安全收尾，当前模式=%s' % self.mode, flush=True)
        if self.mode == 'motion':
            self.motion_safe_zero()
            self.drop_motion()
        elif self.j is not None:
            self.request_release()
            deadline = mono() + RELEASE + 2.0
            while mono() < deadline:
                with self.state_lock:
                    if self.S['sub'] == 'damp':
                        break
                time.sleep(0.05)
            else:
                print('[退出] 渐进放松超时；转阻尼后断开', flush=True)
                with self.state_lock:
                    self.S['sub'] = 'damp'
            time.sleep(0.05)
        self.control_stop.set()
        if self.control_thread is not None:
            self.control_thread.join(timeout=2.0)
        recovery = self.recovery_thread
        if recovery is not None and recovery is not threading.current_thread():
            recovery.join(timeout=2.0)
        if self.rec is not None:
            self.rec.close()
        self.drop_joint()
        self.drop_motion()
        if self.pad is not None:
            self.pad.close()
        print('[退出] 已断开 SDK', flush=True)


def print_help():
    print(r'''
=== BPX 双层手柄（保持原操作）===
MOTION: Y官方站立  A官方趴下  X/RS阻尼  LB普通行走  RB跑步
        趴着时 RB = 切换到 JOINT
JOINT:  Y关节站立并在站稳后自动进策略  A关节趴下  X/RS阻尼
        RB手动补启策略  LB切回MOTION
POLICY: 摇杆控制  LT/RT档位  Y重站并自动进策略  A趴下  X/RS阻尼
START: 安全退出；BACK: 帮助
''', flush=True)


def wait_for_pad(shutdown, retry_interval):
    count = 0
    while not shutdown.is_set():
        path, name = find_pad()
        if path:
            try:
                return Pad(path), name
            except OSError as exc:
                print('[WAIT] 手柄 %s 暂不可打开: %r' % (path, exc), flush=True)
        count += 1
        if count == 1 or count % 10 == 0:
            print('[WAIT] 等待手柄设备…', flush=True)
        shutdown.wait(retry_interval)
    return None, None


def parse_args():
    parser = argparse.ArgumentParser(description='BPX MOTION/JOINT dual-layer gamepad controller')
    parser.add_argument('--ip', default='10.21.20.1', help='机器人 IP')
    parser.add_argument('--gear', type=int, default=2, help='初始档位 1~4')
    parser.add_argument('--passive-delay', type=float, default=1.0, help='趴下完成后开始渐进放松前等待秒数')
    parser.add_argument('--retry-interval', type=float, default=1.5, help='机器人/手柄离线重试间隔')
    parser.add_argument('--no-record', action='store_true', help='do not write log/rec_*.bin (walk recording)')
    return parser.parse_args()


def main():
    args = parse_args()
    app = App(args)

    def request_stop(signum, _frame):
        print('[SIGNAL] 收到 %s，准备安全退出' % signum, flush=True)
        app.shutdown_requested.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    app.start_control_thread()
    try:
        app.pad, pad_name = wait_for_pad(app.shutdown_requested, args.retry_interval)
        if app.pad is None:
            return 1
        print('[PAD] %s (%s)' % (pad_name, app.pad.path), flush=True)
        print_help()
        print('[WAIT] 默认进入 MOTION；可先启动程序，再把唯一网线从电脑换接机器人。', flush=True)
        if not app.connect_motion(attempts=None):
            return 1

        last_status = 0.0
        while not app.shutdown_requested.is_set():
            if not app.pad.update():
                app.on_pad_fault()
                old_pad = app.pad
                app.pad = None
                if old_pad is not None:
                    old_pad.close()
                print('[WAIT] 手柄已断开，进程不退出；等待重新连接…', flush=True)
                app.pad, pad_name = wait_for_pad(app.shutdown_requested, args.retry_interval)
                if app.pad is None:
                    break
                print('[PAD] %s (%s)' % (pad_name, app.pad.path), flush=True)
                app.on_pad_reconnected()
                continue
            app.update_gears()
            button = app.pad.pop()
            while button is not None:
                try:
                    app.handle_button(button)
                except Exception as exc:
                    print('[COMMAND-FAULT] %s: %r' % (BTN_NAME.get(button, hex(button)), exc), flush=True)
                    if app.mode == 'motion':
                        app.motion_safe_zero()
                    elif app.mode in ('joint', 'policy'):
                        app.request_damp('按键处理异常')
                button = app.pad.pop()
            app.maybe_autostart_policy()
            app.motion_tick()
            now = mono()
            if now - last_status >= 2.0:
                last_status = now
                if app.mode == 'motion':
                    print('[状态] MOTION motion=%s gait=%s gear=%d/4 cmd=(%.2f,%.2f,%.2f)' %
                          ((app.motion_state(), app.gait, app.gear + 1) + app.motion_velocity()), flush=True)
                elif app.mode in ('joint', 'policy'):
                    snap = app.snapshot()
                    with app.state_lock:
                        sub = app.S['sub']
                    if snap is None:
                        print('[状态] %s sub=%s 高频状态未就绪' % (app.mode.upper(), sub), flush=True)
                    else:
                        err = float(np.max(np.abs(snap['q'] - STAND_POSE)))
                        print('[状态] %s sub=%s seq=%s age=%.3f 站姿误差=%.3f gear=%d/4' %
                              (app.mode.upper(), sub, snap['seq'], snap['fresh_age'], err, app.gear + 1), flush=True)
                else:
                    print('[状态] %s' % app.mode.upper(), flush=True)
            time.sleep(0.01)
    finally:
        app.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
