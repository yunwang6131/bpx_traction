"""训练损失。

结构沿用 Go2 的 `train_estimator.py`（见 `reference/core_code/training/`），
但有三处按 BPX 实际消费什么改了：

1. **力拆成 xy 和 z 分别加权。** 映射只用 Fx/Fy（`mapping.step()` 取 `force_b[:2]`），
   Fz 估但不用。Go2 的 actor 要吃三维力，所以它三轴等权，结果 Fz 的系统性低估
   （`pred ≈ 0.532×true`）成了大问题。BPX 不吃 Fz，让它少占梯度。
2. **方向损失只看 xy 平面。** 映射的方向完全由 `F_xy` 决定，三维 cosine 里掺进 Fz
   的误差会把真正要紧的平面内方向冲淡。
3. **置信度标定只盯 force 那一维的误差。** 四个置信度里只有 force 有消费端
   （映射用它连续调制斜率 + 硬闸门），其余三个跟着训但不消费。

四个辅助头（速度/高度/触地/负载）**照训**。它们在 BPX 上没有消费端，但那是共享主干的
正则项 —— 参考实现给 height 的权重高达 8.0，删掉会直接改变主干学到的表示。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from . import spec


@dataclass
class LossWeights:
    """各项权重。默认值沿用参考实现，改动的地方标了原因。"""

    velocity: float = 1.0
    height: float = 8.0          # 参考实现就是 8.0，高度比速度难估得多
    contact: float = 1.0
    load: float = 1.0

    force: float = 3.0           # 力这一项的总权重
    force_xy: float = 1.0        # 力内部：水平分量（映射只用这个）
    force_z: float = 0.5         # 力内部：竖直分量（估但不用，少占梯度）
    direction: float = 1.5       # xy 平面内的方向
    calibration: float = 0.5     # 置信度标定

    #: ‖F_true‖ > FORCE_ACTIVE_N 的样本在力项上的额外权重。
    #: 零力样本占四分之一，不加权的话力项会被"预测零"这个平凡解主导。
    active_weight: float = 4.0

    #: 力做 smooth_l1 之前除以这个数。参考实现用 40 —— 让典型误差落在 smooth_l1 的
    #: 二次段（|x|<1）里，而不是线性段。BPX 的设计范围是 0–30 N，用 30 更贴。
    force_norm_n: float = 30.0

    #: 置信度的标定目标 `exp(-decay · err)`。decay 越大，要求越严。
    calibration_decay: float = 3.0


def _slices() -> dict[str, slice]:
    return {name: slice(*spec.output_slice(name)) for name, _ in spec.OUTPUT_TERMS}


def force_losses(
    pred_force: torch.Tensor,
    true_force: torch.Tensor,
    weights: LossWeights,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (幅值项, 方向项, 每样本的力误差)。

    `pred_force` / `true_force` 形状 (B, 3)，单位 N，机身系。
    """
    scale = weights.force_norm_n
    active = torch.linalg.norm(true_force, dim=1) > spec.FORCE_ACTIVE_N

    per_axis = nn.functional.smooth_l1_loss(
        pred_force / scale, true_force / scale, reduction="none"
    )                                                            # (B, 3)
    axis_weight = torch.tensor(
        [weights.force_xy, weights.force_xy, weights.force_z],
        device=pred_force.device, dtype=pred_force.dtype,
    )
    per_sample = (per_axis * axis_weight).sum(dim=1) / axis_weight.sum()

    sample_weight = torch.where(
        active,
        torch.full_like(per_sample, weights.active_weight),
        torch.ones_like(per_sample),
    )
    magnitude_loss = (per_sample * sample_weight).mean()

    # 方向：只看 xy 平面。零力的方向是未定义的，不是"算错了"，所以只在有力的样本上算。
    xy_active = torch.linalg.norm(true_force[:, :2], dim=1) > spec.FORCE_ACTIVE_N
    if bool(xy_active.any()):
        cos = nn.functional.cosine_similarity(
            pred_force[xy_active, :2], true_force[xy_active, :2], dim=1, eps=1e-6
        )
        direction_loss = (1.0 - cos).mean()
    else:
        direction_loss = pred_force.sum() * 0.0          # 保持计算图连通

    # 用于置信度标定的"力误差"：按 force_norm_n 归一的平均绝对误差
    force_error = (pred_force - true_force).abs().mean(dim=1) / scale
    return magnitude_loss, direction_loss, force_error


def estimator_loss(
    values: torch.Tensor,
    confidence: torch.Tensor,
    targets: torch.Tensor,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """总损失 + 各项明细（明细是 float，给日志用）。

    `values` (B, 15)、`confidence` (B, 4)、`targets` (B, 15)。
    """
    w = weights or LossWeights()
    sl = _slices()

    velocity = nn.functional.smooth_l1_loss(values[:, sl["base_lin_vel_b"]],
                                            targets[:, sl["base_lin_vel_b"]])
    height = nn.functional.smooth_l1_loss(values[:, sl["base_height"]],
                                          targets[:, sl["base_height"]])
    # 触地标签是 0/1，预测经过 sigmoid 已在 (0,1)，直接 BCE
    contact = nn.functional.binary_cross_entropy(
        values[:, sl["foot_contact"]].clamp(1e-6, 1 - 1e-6), targets[:, sl["foot_contact"]]
    )
    load = nn.functional.smooth_l1_loss(values[:, sl["foot_load"]], targets[:, sl["foot_load"]])

    magnitude, direction, force_error = force_losses(
        values[:, sl["force_b"]], targets[:, sl["force_b"]], w
    )

    # 置信度标定：目标是 exp(-decay · 误差)，误差大→置信度该低。
    # 四维分别对应 [vel, height, contact, force]。
    errors = torch.stack([
        (values[:, sl["base_lin_vel_b"]] - targets[:, sl["base_lin_vel_b"]]).abs().mean(dim=1),
        (values[:, sl["base_height"]] - targets[:, sl["base_height"]]).abs().mean(dim=1),
        (values[:, sl["foot_contact"]] - targets[:, sl["foot_contact"]]).abs().mean(dim=1),
        force_error,
    ], dim=1)                                                    # (B, 4)
    confidence_target = torch.exp(-w.calibration_decay * errors).clamp(0.0, 1.0).detach()
    calibration = nn.functional.mse_loss(confidence, confidence_target)

    total = (
        w.velocity * velocity
        + w.height * height
        + w.contact * contact
        + w.load * load
        + w.force * magnitude
        + w.direction * direction
        + w.calibration * calibration
    )
    detail = {
        "velocity": float(velocity), "height": float(height), "contact": float(contact),
        "load": float(load), "force": float(magnitude), "direction": float(direction),
        "calibration": float(calibration), "total": float(total),
    }
    return total, detail


# ---------------------------------------------------------------------------
# 批次配比
# ---------------------------------------------------------------------------
#: 按力的大小分桶，再按固定配比抽批。
#:
#: 为什么不直接随机抽：零力样本占 24%，中高力占一半多。随机抽的话，损失会被"把力估小一点
#: 总是安全的"这个方向带偏 —— 而**零力附近恰恰是日常最常见的工况**（机器人大部分时间
#: 没被拉），零力误报会让它无故乱走。
#:
#: 配比沿用参考实现的思路（锚点 40% / 中高力 30% / 低力尾部 15% / 其余 15%），
#: 桶的边界按 BPX 自己的设计范围（0–30 N）定。
FORCE_BUCKETS: list[tuple[str, float, float, float]] = [
    # 名字          下界    上界     配比
    ("zero",         0.0,   2.0,    0.40),   # 锚点：没被拉的时候不许乱报
    ("low_tail",     2.0,   5.0,    0.15),   # 低力方向尾部，最难的一段
    ("mid",          5.0,  15.0,    0.15),
    ("high",        15.0,  30.0,    0.20),
    ("ood",         30.0,  1e9,     0.10),   # 超出设计范围，训置信度说"我不知道"
]


def bucket_indices(targets, buckets=None) -> dict[str, "torch.Tensor"]:
    """按力的幅值把样本分桶，返回每个桶的下标。`targets` 可以是 numpy 或 tensor。"""
    import numpy as np

    arr = targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    begin, end = spec.output_slice("force_b")
    magnitude = np.linalg.norm(arr[:, begin:end], axis=1)
    out = {}
    for name, lo, hi, _ in buckets or FORCE_BUCKETS:
        idx = np.nonzero((magnitude >= lo) & (magnitude < hi))[0]
        out[name] = torch.from_numpy(idx.astype("int64"))
    return out


def sample_balanced_batch(
    buckets: dict[str, torch.Tensor],
    batch_size: int,
    generator: torch.Generator | None = None,
    spec_buckets=None,
) -> torch.Tensor:
    """按 `FORCE_BUCKETS` 的配比抽一个批次的下标。空桶的配额分给其它桶。"""
    plan = spec_buckets or FORCE_BUCKETS
    available = [(name, share) for name, _, _, share in plan if len(buckets.get(name, [])) > 0]
    if not available:
        raise ValueError("所有力桶都是空的 —— 数据集有问题")
    total_share = sum(share for _, share in available)

    picks = []
    for name, share in available:
        pool = buckets[name]
        count = max(1, int(round(batch_size * share / total_share)))
        choice = torch.randint(len(pool), (count,), generator=generator)
        picks.append(pool[choice])
    return torch.cat(picks)[:batch_size]


# ---------------------------------------------------------------------------
# 真机锚定：用无外力录制做域适配
# ---------------------------------------------------------------------------
@dataclass
class RealAnchorWeights:
    """真机那部分损失的权重。

    真机上没有力传感器，但无外力录制里有两个免费标签：**力恒为 0**（采集时没人碰
    机器人），以及**机身速度 ≈ 指令低通后的值**（策略稳态跟踪误差 0.053 m/s）。
    """

    #: 力 → 0。直接打误报，但只有这一项的话网络会退化成"永远输出零"
    #: （真·退化样本：把残差权重装错结构的那个模型仿真 19.25 N，真机看着"误报低"
    #: 其实是完全不响应）。所以仿真批次的力监督必须同时在，权重别压过它。
    force_zero: float = 1.0
    #: 速度 → 低通指令。**这一项才是对因的** —— 真机误报力的慢漂移和
    #: "估计速度 − 指令速度"相关系数 +0.759，是所有候选里最强的；而速度头在真机上
    #: 低估 61%。力头吃的是速度头的残差，把速度头掰正，残差才会小。
    velocity: float = 2.0


def real_anchor_loss(
    values: torch.Tensor,
    vel_label: torch.Tensor,
    vel_weight: torch.Tensor,
    weights: LossWeights | None = None,
    anchor: RealAnchorWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """真机批次的损失。`values` (B,15)、`vel_label` (B,3)、`vel_weight` (B,3)。

    `vel_weight` 为 0 的分量不参与（vz 没有标签；瞬态帧权重打折）。
    """
    w = weights or LossWeights()
    a = anchor or RealAnchorWeights()
    sl = _slices()

    # 力：真值就是零，所以不用 force_losses（它的方向项在零力上没有定义）。
    # 尺度和仿真那边一致，除以 force_norm_n，否则两边的梯度量级对不上。
    pred_force = values[:, sl["force_b"]]
    axis_weight = torch.tensor(
        [w.force_xy, w.force_xy, w.force_z],
        device=pred_force.device, dtype=pred_force.dtype,
    )
    per_axis = nn.functional.smooth_l1_loss(
        pred_force / w.force_norm_n, torch.zeros_like(pred_force), reduction="none"
    )
    force_zero = ((per_axis * axis_weight).sum(dim=1) / axis_weight.sum()).mean()

    pred_vel = values[:, sl["base_lin_vel_b"]]
    per_vel = nn.functional.smooth_l1_loss(pred_vel, vel_label, reduction="none")
    denom = vel_weight.sum().clamp(min=1.0)
    velocity = (per_vel * vel_weight).sum() / denom

    total = a.force_zero * force_zero + a.velocity * velocity
    return total, {"real_force_zero": float(force_zero), "real_velocity": float(velocity)}
