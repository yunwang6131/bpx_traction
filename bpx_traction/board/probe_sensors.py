#!/usr/bin/env python3
"""上 v6 之前先探一下：那两个新 getter 到底有没有、返回什么。

    # 在开发机上启动（板子此时还连着开发机），它会后台等机器人上线
    ./start_probe.sh
    # 然后拔网线插机器人 —— 进程扛得过断链，连上就开始探

和 `start.sh` 一样是**后台等待者**：板子（RK3588）的网线从开发机挪到机器人时 SSH 会断，
但进程还在，它一直重试连接直到机器人上线。

输出**写进文件**（默认 `~/probe_sensors.log`），因为板子插在机器人上的时候开发机
连不上它，根本看不到 stdout。跑完把网线插回开发机取日志。

**纯只读。** 用 `RequestRobotState`（被动订阅状态上报），**一条控制指令都不发**，
不进关节模式、不改控制权、不碰 PD。机器人只要上电站着（或者趴着）就能跑。

## 为什么必须先探

v6 在 `read_high_snapshot()` 里加了两个 getter。如果这台固件上它们不存在或返回
``None``，`vec()` 会抛异常 → 被外面的 ``except Exception: continue`` 吞掉 →
重试三次后整个快照返回 ``None`` → 控制循环报「状态未就绪/超时」。

**如果这发生在机器人走路的时候，就是控制回路断掉。** 所以不能直接把 v6 放上去跑，
先用这个脚本确认接口可用、量纲合理，再部署。

## 探什么

1. `getJointTorque()` 在不在、返回 12 个数、量级合不合理
2. `getImuAcc()` 在不在、模长是不是 ≈9.81（**比力**而不是运动学加速度）
3. `getImuQuat()` 的顺序（低频接口手册标的是 wxyz，和高频版可能不一样）
4. 每次读一轮要多久 —— 500 Hz 的预算是 2000 µs

低频接口能用，高频版（`*HighRate`）基本就没问题 —— 它们读的是同一份状态，
只是推送频率不同。反过来如果低频版就拿不到，那高频版更不用试。
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="只读探测力矩和加速度接口")
    ap.add_argument("--ip", default="10.21.20.1", help="机器人 IP")
    ap.add_argument("--sdk", type=str, default=None,
                    help="bpx_sdk 所在目录，默认按 PYTHONPATH 找")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", type=str, default="~/probe_sensors.log",
                    help="结果写到哪。板子插在机器人上时开发机看不到 stdout，"
                         "所以必须落盘")
    ap.add_argument("--rate", type=int, default=100, help="状态上报频率 Hz")
    ap.add_argument("--wait-s", dest="wait_s", type=float, default=600.0,
                    help="最多等机器人上线多久。默认 10 分钟 —— 够你拔线插线了")
    args = ap.parse_args()

    log_path = Path(args.out).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")

    def say(*parts, err=False):
        text = " ".join(str(x) for x in parts)
        print(text, file=sys.stderr if err else sys.stdout, flush=True)
        log.write(text + "\n")
        log.flush()

    say(f"# 探测时间 {time.strftime('%Y-%m-%d %H:%M:%S')}   机器人 {args.ip}")

    if args.sdk:
        sys.path.insert(0, str(Path(args.sdk).expanduser().resolve()))
    try:
        import bpx_sdk
    except ImportError:
        say("导入不了 bpx_sdk —— 用 --sdk 指定它所在的目录", err=True)
        return 1

    state = bpx_sdk.RequestRobotState()
    state.setRobotIp(args.ip)
    state.setRobotStateUploadRate(args.rate)
    # 等机器人上线。板子的网线从开发机挪到机器人要几十秒，这期间连不上是正常的。
    say(f"等机器人 {args.ip} 上线…（只读，不发任何控制指令；最多等 {args.wait_s:.0f} s）")
    say("现在可以拔掉开发机网线，把 eth0 接到机器人了。")
    deadline = time.monotonic() + args.wait_s
    attempt = 0
    while True:
        if state.connect():
            break
        attempt += 1
        if time.monotonic() > deadline:
            say(f"等了 {args.wait_s:.0f} s 还没连上（试了 {attempt} 次）。"
                "检查网线和机器人电源。", err=True)
            return 1
        if attempt % 10 == 1:
            say(f"  还没连上，继续等…（第 {attempt} 次）")
        time.sleep(2.0)
    say(f"连上了（等了 {attempt} 次重试）")

    try:
        time.sleep(0.5)                       # 等第一帧状态上来

        results: dict[str, list] = {"tau": [], "acc": [], "quat": [], "us": []}
        deadline = time.monotonic() + args.seconds
        missing = {"tau": 0, "acc": 0, "quat": 0}

        while time.monotonic() < deadline:
            t0 = time.monotonic()
            tau = state.getJointTorque()
            acc = state.getImuAcc()
            quat = state.getImuQuat()
            results["us"].append((time.monotonic() - t0) * 1e6)

            for key, value in (("tau", tau), ("acc", acc), ("quat", quat)):
                if value is None:
                    missing[key] += 1
                else:
                    results[key].append(list(value))
            time.sleep(1.0 / args.rate)

        say()
        ok = True

        # ---- 力矩 ----
        if not results["tau"]:
            say(f"✗ getJointTorque() 一次都没返回（{missing['tau']} 次 None）")
            say("  这台固件不支持力矩读数 —— 力估计器的 12 维力矩输入在真机上拿不到，")
            say("  整条路要重新设计输入。**先别部署 v6。**")
            ok = False
        else:
            cols = list(zip(*results["tau"]))
            means = [statistics.fmean(c) for c in cols]
            peak = max(abs(v) for row in results["tau"] for v in row)
            say(f"✓ getJointTorque()   {len(results['tau'])} 帧，{len(cols)} 个关节"
                  f"（None {missing['tau']} 次）")
            say(f"  均值  roll  {[round(means[i], 2) for i in range(0, 12, 3)]}")
            say(f"        pitch {[round(means[i], 2) for i in range(1, 12, 3)]}")
            say(f"        knee  {[round(means[i], 2) for i in range(2, 12, 3)]}")
            say(f"  |τ| 峰值 {peak:.2f} N·m")
            knee = statistics.fmean(abs(means[i]) for i in range(2, 12, 3))
            if knee < 0.5:
                say("  ⚠️ 膝关节均值接近 0。站立时膝要撑体重，不该这么小 ——")
                say("     读数可能是前馈指令而不是测量值。趴着测的话忽略这条。")

        # ---- 加速度 ----
        if not results["acc"]:
            say(f"✗ getImuAcc() 一次都没返回（{missing['acc']} 次 None）")
            ok = False
        else:
            cols = list(zip(*results["acc"]))
            means = [statistics.fmean(c) for c in cols]
            norm = math.sqrt(sum(v * v for v in means))
            say(f"\n✓ getImuAcc()        均值 {[round(v, 2) for v in means]}   "
                  f"模长 {norm:.2f} m/s²")
            if abs(norm - 9.81) < 2.0:
                say("  ✓ 模长 ≈9.81 → 是**比力**（含重力），和仿真的 "
                      "ImuCfg(gravity_bias=(0,0,9.81)) 一致")
            elif norm < 3.0:
                say("  ⚠️ 模长接近 0 → 是**运动学加速度**，不含重力。")
                say("     仿真那边必须把 gravity_bias 改成 (0,0,0) 重采重训，否则 sim2real 直接废。")
            else:
                say(f"  ⚠️ 模长 {norm:.2f}，既不是 0 也不是 9.81 —— 查单位")

        # ---- 四元数顺序 ----
        if results["quat"]:
            cols = list(zip(*results["quat"]))
            mean_abs = [statistics.fmean(abs(v) for v in c) for c in cols]
            order = sorted(range(4), key=lambda i: mean_abs[i])[:2]
            say(f"\n  getImuQuat()       |各位均值| {[round(v, 3) for v in mean_abs]}")
            if set(order) == {0, 1}:
                say("  → 接近零的在第 1、2 位（= roll/pitch 分量 x,y）⇒ **xyzw**")
            elif set(order) == {1, 2}:
                say("  → 接近零的在第 2、3 位 ⇒ **wxyz**（和手册 6.3 一致）")
            else:
                say("  → 判别不了，机器人当时倾角太大？站平再测")
            say("     注意这是**低频**接口。板子用的是 getImuQuatHighRate()，"
                  "两者顺序不一定相同。")

        # ---- 时序 ----
        us = sorted(results["us"])
        if us:
            p50 = us[len(us) // 2]
            p99 = us[min(len(us) - 1, int(len(us) * 0.99))]
            say(f"\n  读一轮（3 个 getter） p50 {p50:.0f}   p99 {p99:.0f} µs")
            say(f"     500 Hz 的预算是 2000 µs/步。这里是低频接口的量级，"
                  f"高频版应当更快。")

        say()
        if ok:
            say("两个接口都可用 → 可以部署 v6 采数据了。")
        else:
            say("有接口拿不到 → **先别部署 v6**，拿着上面的输出我们再定怎么改。")
        return 0 if ok else 2

    finally:
        say(f"\n（这份输出也写在 {log_path} 了）")
        log.close()
        try:
            state.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
