"""训练时的传感器增广 —— 补仿真传感器模型和真机的差距。

## 为什么需要

IsaacLab 的 `ImuSensor` 是对机身速度做有限差分，刚性接触的冲量会产生巨大的瞬时
加速度；真 IMU 既有带宽限制，机器人也有橡胶脚垫和机架柔性把冲击吸掉。实测
（走路、无外力）：

    acc_xy      仿真 p50 11.15  p95 28.83  p99 36.49  max 63.00 m/s²
                真机 p50  1.89  p95  5.96  p99  9.72  max 23.82 m/s²

**整整差 6 倍，而且是全频段的** —— 不是带宽问题：仿真过 2 Hz 低通之后 p95 还有
11.91，真机只有 5.96；高频能量占比反而是真机更高（50.5% vs 30.8%，因为真机信号
本身小，噪声占比大）。根因是接触刚度和柔性，而柔性不在执行器辨识的范围里。

## 为什么这件事要紧

加速度是估计器第二重要的输入（消融：置零 → xy 误差 +125%，仅次于力矩的 +351%）。
把真机的 acc_xy **放大**到仿真量级之后，直接验证过因果关系：

    放大倍数 k     速度头/指令     误报 p95
        1.0          0.39        32.22 N
        6.0          0.53        20.82 N      ← 降 35%

即网络学的就是仿真那个分布：真机读数小 6 倍，它就当成"没在动、没被拉"。
速度头低估 61% 和力误报是同一个原因的两个表现。

## 怎么修

**标定，不是随机化。** 比值在所有活动水平下都稳定在 ~5 倍（静止 4.5× / 慢走 5.2× /
走 5.7× / 快走 5.8×），所以这是个可标定的系统性偏差，不是不确定性。固定缩放
0.2× 之后两边的 p50 和 p95 在各个活动档上都对得上。

走过的弯路：第一版把区间设成 (0.15, 1.2)，想法是"覆盖两边让网络不敢依赖尺度"。
实测的结果是网络干脆不用这个通道了 —— 速度头确实修好了（0.39 → 0.59），
但**力误报纹丝不动**（32.22 → 31.38），仿真 xy 和高力段各退 10%。

区间保留 ±40% 是留给机器人之间的差异，不是用来跨越 sim2real 鸿沟的。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import spec


@dataclass
class ImuAugment:
    """加速度通道的增广参数。`enabled=False` 时 `apply` 原样返回。

    默认值是**把仿真标定到真机**，不是覆盖两边 —— 理由见模块文档"怎么修"那节。
    """

    enabled: bool = False
    #: 每个样本的乘性缩放，只作用在 **xy** 两轴上（z 轴主要是重力，动了就破坏比力契约）。
    #:
    #: **默认是标定值不是随机化区间。** 实测仿真/真机的比值在所有活动水平下都是
    #: ~5 倍（静止 4.5×、慢走 5.2×、走 5.7×、快走 5.8×），所以 0.2× 能把两边对齐：
    #:
    #:     ‖acc_xy‖     静止p50  慢走p50  走p50  慢走p95
    #:     仿真 × 0.2     0.134    1.50    1.76    4.06
    #:     真机实测       0.15     1.45    1.53    4.06
    #:
    #: 区间 (0.15, 0.28) = 0.2 上下 ±40%，留出机器人之间的差异。
    #:
    #: ⚠ **不要**把区间放宽成 (0.15, 1.2) 那种"覆盖两边"的做法 —— 试过，那是教网络
    #: 对尺度不敏感，结果它干脆不用这个通道了：速度头从 0.39 修到 0.59，但力误报
    #: 纹丝不动（32.22 → 31.38），仿真指标还退了 10%。要的是**标定**不是**忽略**。
    scale_range: tuple[float, float] = (0.15, 0.28)
    #: 一阶低通的截止频率区间（Hz）。上界 25 Hz 是奈奎斯特，等于"几乎不滤"。
    #: 实测低通**单独**解释不了那 6 倍差距（仿真滤到 2 Hz 后 p95 还有 11.91，
    #: 真机只有 5.96），所以主力是上面的缩放，低通只负责抹掉刚性接触的尖峰。
    lpf_hz_range: tuple[float, float] = (4.0, 25.0)
    #: 加性噪声的标准差（m/s²）。**真机实测本底**：静止 78 s（|dq|<0.05 rad/s）
    #: 上 acc_x σ=0.086、acc_y σ=0.070，取 0.08。
    #:
    #: ⚠ 顺序要紧：噪声加在**缩放之前**。先缩放再加噪的话，0.2× 之后信号 p50 只剩
    #: 1.5，固定的噪声就成了 20% 的相对噪声（不缩放时只有 4%）—— 等于把信噪比
    #: 砍掉 5 倍。第一轮标定实验就是栽在这儿：仿真 xy 从 4.535 翻到 9.083，
    #: 看着像"标定有害"，其实是噪声注错了位置。
    noise_std: float = 0.08

    def describe(self) -> str:
        if not self.enabled:
            return "关"
        return (f"xy 缩放 {self.scale_range}  低通 {self.lpf_hz_range} Hz  "
                f"噪声 σ={self.noise_std:g} m/s²")


def _lowpass(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """沿时间轴的一阶低通。`x` (B, T, C)，`alpha` (B, 1, 1)。

    用第一帧做初值，不是零 —— 零初值会在窗口开头造出一段假的上升沿，
    而最旧的那几帧恰恰是空洞卷积最敏感的位置（gridding，实测敏感度是最新帧的 4 倍）。
    """
    out = torch.empty_like(x)
    out[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        out[:, t] = alpha[:, 0] * out[:, t - 1] + (1.0 - alpha[:, 0]) * x[:, t]
    return out


def apply_imu_augment(history: torch.Tensor, cfg: ImuAugment,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """对 **未归一化** 的 (B, T, 61) 历史做增广，返回新张量。

    必须在归一化之前调用 —— 增广改的是物理量，归一化是之后的事。
    """
    if not cfg.enabled:
        return history

    begin, _end = spec.term_slice("imu_acc_b")
    xy = history[:, :, begin:begin + 2]
    b = history.shape[0]
    dev = history.device
    rnd = lambda lo, hi: torch.empty(b, 1, 1, device=dev).uniform_(lo, hi, generator=generator)

    out = _lowpass(xy, torch.exp(-2.0 * torch.pi * rnd(*cfg.lpf_hz_range) * spec.SAMPLE_PERIOD_S))
    if cfg.noise_std > 0:
        # 先加噪再缩放 —— 噪声是传感器在**真机尺度**上的本底，不是仿真尺度上的
        out = out + torch.randn(out.shape, device=dev, generator=generator) * (
            cfg.noise_std / max(sum(cfg.scale_range) / 2.0, 1e-6))
    out = out * rnd(*cfg.scale_range)

    history = history.clone()
    history[:, :, begin:begin + 2] = out
    return history
