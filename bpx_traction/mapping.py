"""力 → 速度指令：一阶导纳。

这一层是**非学习**的，纯 numpy，不依赖 torch / isaaclab，板子上直接跑。

## 为什么是导纳控制

整条链路是标准的导纳（admittance）结构：

    力感知 → 外环（导纳）→ 速度指令 → 刚性内环（速度跟踪策略 + PD）

和教科书导纳控制的区别有三条，都是实质性的：

1. **力是估计的，不是测量的。** F/T 传感器是 kHz 级、误差 <1%；这里的估计器感受野
   180 ms、误差 1 N 量级。所以**导纳外环的带宽必须远低于力感知带宽**，否则估计噪声
   会被放大成真实运动。下面的 ``virtual_mass_kg`` 就是在买这个带宽裕度。
2. **内环不是伺服，是学出来的策略。** 它的跟踪特性只有实测曲线，没有传递函数，
   而且它还要同时维持平衡。
3. **没有刚度项。** 经典导纳常带虚拟弹簧 ``K·∫v``（被拉走会想回去），牵引不需要 ——
   被拉走就是要走。所以是纯 M–B 的一阶系统。

## 参数为什么用 M / B 而不是增益和时间常数

    M · v̇ + B · v = F

    B = 虚拟阻尼 [N·s/m]   稳态 v = F / B —— 拉着它以 1 m/s 走，手上感觉到 B 牛
    M = 虚拟质量 [kg]      时间常数 τ = M / B —— 感觉这东西有多"沉"

这两个数能用手感直接判断（"太沉了"减 M，"太黏了"减 B，"一拉就窜"加 B），
而等价的"增益 K = 1/B、低通 τ"是两个拍脑袋的数。标定从凑参数变成调手感。

默认 M=3.2 kg / B=17.8 N·s/m：拉着它以 1 m/s 走手上约 18 N，虚拟质量只有整机 15.1 kg 的
两成 —— 这正是牵引想要的，**拉起来比它本身轻**。B 是从实测的 30 N 饱和点反解的，见下。

## 为什么有 min_speed

`bpx_rma.mdp.gait.gait_phase_obs` 在 ``‖cmd_xy‖ + |cmd_yaw| ≤ 0.1`` 时把相位观测置零
（"不走就没有节拍"）。牵引层的输出会**连续地**穿过这个阈值 —— 力慢慢变大，指令从 0
爬到 0.05、0.09、0.11…… 相位就在阈值附近反复开关，策略看到的节拍一会儿有一会儿没有。

所以死区外的第一个非零输出直接跳到 ``min_speed``（默认 0.15 m/s），跳过 0–0.1 这个带。
配合 Schmitt 死区的迟滞，力在边界附近抖动也不会让指令来回开关。

只有走相位进观测那条线（Sprint6 及以后）时这条才是必需的，但 45 维基底下加上也无害 ——
人拉动它本来也该有个最小启动速度，否则看起来像在蠕动。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import spec

DEFAULT_PARAMS_PATH = spec.PACKAGE_DIR / "data" / "admittance.json"


@dataclass
class AdmittanceParams:
    """导纳 + 死区 + 限幅的全部可调量。

    标 ``# 待实测`` 的三个必须用数据定，现在的值是占位，不要当成标定结果。
    """

    # --- 导纳本体 ---
    virtual_mass_kg: float = 3.2          # M。τ = M/B ≈ 0.18 s
    #: B。**不要手填** —— 由 `damping_for_saturation(30, 1.5, 6.0, 0.15)` 反解得到：
    #: "力到 30 N 时指令正好到 vx 上限 1.5"。30 N 是扛力扫描实测的策略上限
    #: （见 `spec.FORCE_DESIGN_RANGE_N`），再往上机器人会被拖走，发再大的指令也没用。
    #: 死区实测出来之后，用同一个函数重算，不要回头凑数。
    virtual_damping_ns_per_m: float = 17.77777777777778

    # --- 死区（Schmitt 迟滞，避免边界抖动）---
    deadzone_on_n: float = 8.0            # 仍是占位，**不是**标定结果
    deadzone_off_n: float = 6.0           # 同上
    # 占位值来自 Go2 的跟随阈值（‖F_xy‖ ≥ 8 N 必须跟随、< 8 N 必须站住）。
    #
    # 2026-10-26 用真机量过两轮，结论都是**现有估计器定不出可用的死区**，所以占位值
    # 没动、`calibrated` 保持 False。数据：300 s 真机无外力行走（sprint7，全程没人碰
    # 机器人，所以估计器吐的任何非零力都是误报），`scripts/real_deadzone.py --rec
    # logs/real_v6/rec_20260418_191052`。
    #
    # 第一轮，纯仿真训练的模型：
    #
    #     模型                 仿真 xy RMSE   真机误报 p99    max     15N 误起步/分
    #     estimator_dilated       4.535        38.70 N     49.91 N      21.9
    #     estimator_residual      4.615        45.59 N     53.50 N      25.3
    #
    # 第二轮，用真机无外力数据做过域适配（`scripts/finetune_real.py`）。下面是
    # **时间留出法**（前 70% 训、后 30% 只评估）在留出段上的零误起步阈值：
    #
    #     去抖      微调前       微调后
    #     0.2 s    >40 N       40 N
    #     0.3 s    >40 N       35 N
    #     0.5 s     40 N       30 N
    #
    # 设计力范围上限就是 30 N（`spec.FORCE_DESIGN_RANGE_N`，扛力扫描实测，再大机器人
    # 直接被拖走）。微调把需求从"高出天花板 33%"压到"正好顶在天花板上"，**可用区间
    # 仍然是零**。
    #
    # 瓶颈是真机数据量，不是训练：样本外指标在 epoch 32 就平了（22.4–23.2 无趋势），
    # 而样本内还在降 —— 典型的过拟合签名。只有一次录制、一种地面、一只机器人。
    # 下一步是补采真机数据，不是接着训。
    #
    # 已经确定下来的一条性质：**误报力不随速度变化**（前进 0.05–0.6 / 0.6–1.2 / >1.2
    # 三段 p99 几乎相同），所以死区是个常数，不用做成速度的函数。
    #
    # 还有一条约束供重标时参照：真机走路时误报力 p50 已经 8 N，高过现在的
    # `deadzone_off_n`。`release_hold_s = 0` 是掉到 off 以下立刻停，off 设得比噪声
    # 低的话，一旦起步机器人自己的步态噪声就会把它顶住，永远松不开。

    # --- 持续时间：区分"被推了一下"和"被牵着走" ---
    #: 力必须**连续**超过 `deadzone_on_n` 这么久才起步。
    #:
    #: 推搡是瞬态的（人推一把，接触大概 100–200 ms），牵引是持续的（秒级）。
    #: 两者在**瞬时力**上长得一模一样，区别只在时间尺度上 —— 所以靠时长判，这是唯一
    #: 能分开它们的信号。没有这道去抖，机器人会被每一次碰撞、每一次踩空的反作用、
    #: 每一次估计器误报拽着走。
    #:
    #: 代价是延迟。总的起步延迟大致是：
    #:     估计器感受野 180 ms + 去抖 engage_hold_s + 导纳 τ
    #: 默认 0.2 s 下合计约 0.5 s，正好压在响应预算上。调大它更不容易误触发，
    #: 但人拉绳子之后要等更久才动。
    engage_hold_s: float = 0.20
    #: 反过来：力掉到 `deadzone_off_n` 以下多久才停。设成 0 = 立刻停（更安全）。
    release_hold_s: float = 0.0

    # --- 跳过 gait_phase_obs 的 stand_threshold ---
    min_speed_mps: float = 0.15
    stand_threshold_mps: float = 0.1      # 对应 gait.gait_phase_obs 的同名参数

    # --- 安全 ---
    max_accel_mps2: float = 2.0           # 指令变化率上限
    #: 置信度低于它开始计数，连续 `conf_gate_steps` 步就闸死。
    #:
    #: ⚠ **这道闸在真机上是空的。** 2026-10-26 实测（300 s 行走）：力置信度
    #: p50 0.871 / p95 0.883，**14821/14821 个样本全部高于 0.55**，包括那些报了
    #: 50 N 其实没有外力的样本 —— 它一边报错一边说自己可信。
    #:
    #: 原因是这个头只在仿真数据上学过"什么情况下我误差小"（`losses.py` 的
    #: `confidence_target = exp(-3·误差)`）。在仿真里它是标定好的（秩相关 −1.00，
    #: 置信度 0.69 对应误差 10.3 N、0.90 对应 2.31 N），到真机输入看起来仍在分布内
    #: （各列 z 分数都在 ±3 内）误差却大一个量级，它没有能力察觉。
    #: 调阈值解决不了：要么训练时喂分布外负样本，要么换一种不靠网络自评的闸门。
    conf_gate: float = 0.55
    conf_gate_steps: int = 10             # 连续这么多步（200 ms @50Hz）就闸死

    # --- 限幅：必须夹在**策略训练分布**内，不是板子限幅 ---
    vx_range: tuple[float, float] = (-1.0, 1.5)
    vy_range: tuple[float, float] = (-0.5, 0.5)
    # 板子的 POLICY_VX_FWD_MAX = 3.0 / POLICY_VY_MAX = 0.5 是手柄能发出的上限，
    # 和策略见过的指令分布是两回事。发一个训练分布外的指令，策略的行为无定义。
    # 这两个范围由 check_traction_contract.py 对当前 env cfg 的
    # commands.base_velocity.ranges 断言，改了那边这里会报错。

    # --- 来源标注 ---
    source_task: str = "Isaac-Velocity-Flat-BpxPro-v0"   # 限幅抄自哪个任务
    calibrated: bool = False              # 真机标定过之后改 True

    def tau_s(self) -> float:
        """一阶时间常数 M / B。"""
        return self.virtual_mass_kg / self.virtual_damping_ns_per_m

    @staticmethod
    def damping_for_saturation(
        saturation_force_n: float,
        max_speed_mps: float,
        deadzone_off_n: float,
        min_speed_mps: float,
    ) -> float:
        """反解 B：要让"力到 ``saturation_force_n`` 时指令正好到上限"，B 该是多少。

        ``B = (F_sat − 死区) / (v_max − v_min)``

        **B 不该手填。** 它由三个有依据的量决定：死区（实测走路时的机身力波动）、
        速度上限（策略的训练分布）、起步速度（gait 的 stand_threshold）。死区一旦测出来，
        B 跟着重算，而不是回头再凑一个数。

        >>> AdmittanceParams.damping_for_saturation(40.0, 1.5, 6.0, 0.15)
        25.185...
        """
        span = max_speed_mps - min_speed_mps
        if span <= 0:
            raise ValueError(f"v_max({max_speed_mps}) 必须大于 v_min({min_speed_mps})")
        if saturation_force_n <= deadzone_off_n:
            raise ValueError(
                f"饱和力({saturation_force_n}) 必须大于死区({deadzone_off_n})"
            )
        return (saturation_force_n - deadzone_off_n) / span

    def saturation_force_n(self) -> float:
        """当前参数下，指令达到 vx 上限所需的力。和设计意图对不上就调 B。"""
        return (
            self.deadzone_off_n
            + (self.vx_range[1] - self.min_speed_mps) * self.virtual_damping_ns_per_m
        )

    def steady_speed(self, force_n: float, confidence: float = 1.0) -> float:
        """给定恒力下的稳态速度（未限幅）。用来反推 B 该取多少。"""
        if force_n <= self.deadzone_off_n:
            return 0.0
        return self.min_speed_mps + confidence * (
            force_n - self.deadzone_off_n
        ) / self.virtual_damping_ns_per_m

    def validate(self) -> None:
        if not self.deadzone_off_n <= self.deadzone_on_n:
            raise ValueError(
                f"Schmitt 死区要 off({self.deadzone_off_n}) ≤ on({self.deadzone_on_n})，"
                "否则没有迟滞"
            )
        if self.min_speed_mps <= self.stand_threshold_mps:
            raise ValueError(
                f"min_speed({self.min_speed_mps}) 必须大于 stand_threshold"
                f"({self.stand_threshold_mps})，否则相位观测会在阈值附近抖"
            )
        if self.virtual_mass_kg <= 0 or self.virtual_damping_ns_per_m <= 0:
            raise ValueError("M 和 B 必须为正")
        for name, (lo, hi) in (("vx", self.vx_range), ("vy", self.vy_range)):
            if lo >= hi:
                raise ValueError(f"{name}_range 顺序反了：{(lo, hi)}")

    def to_json(self, path: str | Path = DEFAULT_PARAMS_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_PARAMS_PATH) -> "AdmittanceParams":
        data = json.loads(Path(path).read_text())
        for key in ("vx_range", "vy_range"):
            if key in data:
                data[key] = tuple(data[key])
        obj = cls(**data)
        obj.validate()
        return obj


@dataclass
class MappingState:
    """导纳的全部状态。`reset()` 之后和冷启动等价。"""

    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    engaged: bool = False          # Schmitt 触发器的状态
    above_steps: int = 0           # 连续超过 deadzone_on 的步数（去抖用）
    below_steps: int = 0           # 连续低于 deadzone_off 的步数
    low_conf_steps: int = 0        # 连续低置信度的步数
    bias_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))


class AdmittanceMapping:
    """力 → ``(vx, vy, 0.0)``。

    用法::

        mapping = AdmittanceMapping(AdmittanceParams.from_json())
        mapping.calibrate_bias(static_force_samples)      # 开机静止 5 s
        for each control step:
            cmd = mapping.step(force_b, confidence_force)  # -> (3,)

    `wz` 恒为 0：牵引只影响 xy 平面。被侧拉时机器人保持朝向、横向移动，
    而不是转身迎上去。代价是 vy 上限只有 0.5 m/s（vx 是 1.5），90° 侧拉跟不上
    人走路的速度 —— 这是已知的、有意的限制，不是 bug。
    """

    def __init__(self, params: AdmittanceParams | None = None, dt: float | None = None):
        self.params = params or AdmittanceParams()
        self.params.validate()
        self.dt = spec.SAMPLE_PERIOD_S if dt is None else dt
        self.state = MappingState()

    # -- 标定 --------------------------------------------------------------
    def calibrate_bias(self, force_samples: np.ndarray) -> np.ndarray:
        """开机静止时采一段 F̂，取均值当零偏。``force_samples`` 形状 (N, 3) 或 (N, 2)。"""
        samples = np.asarray(force_samples, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[1] < 2:
            raise ValueError(f"期望 (N, 2) 或 (N, 3)，拿到 {samples.shape}")
        self.state.bias_xy = samples[:, :2].mean(axis=0)
        return self.state.bias_xy

    def reset(self) -> None:
        """清掉滤波器和触发器状态，保留已标定的零偏。"""
        bias = self.state.bias_xy
        self.state = MappingState(bias_xy=bias)

    # -- 主循环 ------------------------------------------------------------
    def step(self, force_b, confidence_force: float, dt: float | None = None) -> np.ndarray:
        """走一个控制步，返回 ``(vx, vy, 0.0)``。

        Args:
            force_b: 机身系外力 (3,) 或 (2,)，单位 N。Fz 不参与映射。
            confidence_force: 力估计的置信度 ∈ [0, 1]（估计器输出的第 4 维）。
            dt: 步长，默认用构造时的。
        """
        p = self.params
        s = self.state
        dt = self.dt if dt is None else dt

        force = np.asarray(force_b, dtype=np.float64).reshape(-1)
        f_xy = force[:2] - s.bias_xy
        magnitude = float(np.linalg.norm(f_xy))
        conf = float(np.clip(confidence_force, 0.0, 1.0))

        # 1) 置信度硬闸门：连续低置信度才闸，避免单帧抖动就停机
        s.low_conf_steps = s.low_conf_steps + 1 if conf < p.conf_gate else 0
        gated = s.low_conf_steps >= p.conf_gate_steps

        # 2) Schmitt 死区 + 持续时间去抖
        #    瞬时力分不出"被推了一下"和"被牵着走"，只有时长能分。所以起步要求力
        #    **连续**超过 deadzone_on 够久；一旦起步，用更低的 deadzone_off 维持（迟滞）。
        engage_steps = max(1, int(round(p.engage_hold_s / dt))) if p.engage_hold_s > 0 else 1
        release_steps = max(1, int(round(p.release_hold_s / dt))) if p.release_hold_s > 0 else 1

        if s.engaged:
            s.above_steps = 0
            s.below_steps = s.below_steps + 1 if magnitude <= p.deadzone_off_n else 0
            if s.below_steps >= release_steps:
                s.engaged = False
                s.below_steps = 0
        else:
            s.below_steps = 0
            s.above_steps = s.above_steps + 1 if magnitude > p.deadzone_on_n else 0
            if s.above_steps >= engage_steps:
                s.engaged = True
                s.above_steps = 0

        # 3) 稳态目标速度
        if gated or not s.engaged or magnitude <= 1e-9:
            target = np.zeros(2)
        else:
            # 置信度只调斜率，不调 min_speed —— 否则低置信度时输出会掉回
            # stand_threshold 以下，相位观测又开始抖
            speed = p.min_speed_mps + conf * (magnitude - p.deadzone_off_n) / p.virtual_damping_ns_per_m
            target = speed * (f_xy / magnitude)       # 方向完全由力决定

        # 4) 一阶导纳。零阶保持的精确离散，无条件稳定（显式欧拉在 dt·B/M 大时会炸）
        alpha = math.exp(-dt / p.tau_s())
        velocity = alpha * s.velocity + (1.0 - alpha) * target

        # 5) 速率限幅：硬上限，防止指令阶跃把策略打崩
        delta = velocity - s.velocity
        max_delta = p.max_accel_mps2 * dt
        norm_delta = float(np.linalg.norm(delta))
        if norm_delta > max_delta:
            velocity = s.velocity + delta * (max_delta / norm_delta)

        # 6) 限幅到训练分布。各向同性的增益保证了方向不失真，这里按轴夹是物理限制
        velocity = np.array([
            np.clip(velocity[0], *p.vx_range),
            np.clip(velocity[1], *p.vy_range),
        ])

        # 状态存的是**实际输出**，不是中间量 —— 否则滤波器会对着一个夹不住的目标积分
        s.velocity = velocity
        return np.array([velocity[0], velocity[1], 0.0])

    # -- 诊断 --------------------------------------------------------------
    def describe(self) -> str:
        p = self.params
        lines = [
            f"一阶导纳   M = {p.virtual_mass_kg:.2f} kg   B = {p.virtual_damping_ns_per_m:.1f} N·s/m"
            f"   τ = {p.tau_s() * 1000:.0f} ms",
            f"           整机 15.1 kg，虚拟质量是它的 {p.virtual_mass_kg / 15.1:.0%}"
            f"（拉起来应该比实际轻）",
            f"死区       进 {p.deadzone_on_n:.1f} N / 出 {p.deadzone_off_n:.1f} N"
            f"{'' if p.calibrated else '   ⚠️ 占位值，待实测'}",
            f"去抖       力要连续超过死区 {p.engage_hold_s * 1000:.0f} ms 才起步"
            f"（推一下 100–200 ms，拒掉；牵引是秒级，放行）",
            f"起步延迟   感受野 180 + 去抖 {p.engage_hold_s * 1000:.0f} + 导纳 τ "
            f"{p.tau_s() * 1000:.0f} ≈ {180 + p.engage_hold_s * 1000 + p.tau_s() * 1000:.0f} ms",
            f"起步速度   {p.min_speed_mps:.2f} m/s（跳过 stand_threshold {p.stand_threshold_mps:.2f}）",
            f"限幅       vx {p.vx_range}   vy {p.vy_range}   wz 恒 0",
            f"           来源 {p.source_task}",
            f"饱和力     {p.saturation_force_n():.1f} N 时指令到 vx 上限"
            f"（设计范围上限 {spec.FORCE_DESIGN_RANGE_N[1]:.0f} N）",
            f"闸门       conf < {p.conf_gate} 连续 {p.conf_gate_steps} 步"
            f"（{p.conf_gate_steps * spec.SAMPLE_PERIOD_S * 1000:.0f} ms）→ 指令归零",
            "",
            "稳态响应（置信度 1.0）：",
        ]
        for f_n in (0, 5, 8, 10, 15, 20, 30, 40):
            v = p.steady_speed(f_n)
            capped = min(v, p.vx_range[1])
            mark = "  ← 限幅" if v > capped else ""
            lines.append(f"    {f_n:3d} N → {capped:.3f} m/s{mark}")
        return "\n".join(lines)


if __name__ == "__main__":
    m = AdmittanceMapping()
    print(m.describe())
