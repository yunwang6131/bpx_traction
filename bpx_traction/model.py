"""力/状态估计器：逐帧编码 + 因果 TCN + 双头。

结构还原自 Go2 交付包的 TorchScript（`missing_state_world_v2_estimator.ts`，该包已删除，
还原记录见 `reference/PROVENANCE.md` §2.1）：

    输入 (B, 25, D) → 归一化 → transpose → (B, D, 25)
      Conv1d(D → C, kernel=1)                        逐帧线性编码
      4 × [Conv1d(C → C, kernel=3) + Norm + 激活]     TCN
      取最后一帧 → (B, C)
      Linear(C → 15)  values
      Linear(C → 4)   confidence → sigmoid

和原件的两处**有意不同**：

1. **因果 padding。** 原件的 TorchScript 看不出 padding 方式。这里只在左侧补零。
   因为前向取的是最后一帧（`select(-1)`），若用 same padding，最后一帧会卷到右侧补的零，
   等于"看到了未来是零"—— 训练时这是个一致的假象，部署时同样成立，所以不会崩，
   但它让感受野的含义变得含糊。左 padding 下每一帧只看自己和更早的帧，干净。

2. **归一化方式可选**（`NORMS`），默认 `"layer"` = 逐帧 LayerNorm，在通道维上归一、不跨时间。

   `"group"` 用 `GroupNorm(4, C)`，统计量跨通道组**和整个时间轴**池化。这会让窗口内
   第 t 帧的中间表示依赖到更晚的帧 —— 但**部署时仍然是因果的**：窗口的末端就是"现在"，
   整个窗口都是过去，而前向只读最后一帧（`x[:, :, -1]`），不存在未来的数据参与计算。

   代价是两条：`receptive_field()` 这个说法对 group 不再成立（输出通过归一化统计量
   依赖全部 25 帧，和卷积够不够得着无关）；以及环形缓冲刚灌满时（25 份同一帧复制）
   时间方向上方差为零，统计量和稳态差很远，起步那一瞬的输出要单独盯。

输出头的激活和量程全部从 :mod:`bpx_traction.spec` 读，模型不能和契约走散。
"""

from __future__ import annotations

import torch
from torch import nn

from . import spec


class ChannelLayerNorm(nn.Module):
    """(B, C, T) 上按通道维归一化，逐帧独立。参数是 (C,) 的 weight / bias。"""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


#: 可选的归一化方式。差别**只在统计量沿哪些维度池化**，这决定了网络对
#: "整个窗口共同的慢偏移"敏不敏感：
#:
#: * ``"layer"``：:class:`ChannelLayerNorm`，逐帧在通道维上归一。每一帧各算各的，
#:   窗口内所有帧共有的偏移会**原样传下去**。
#: * ``"group"``：``nn.GroupNorm(4, C)``，统计量跨 16 个通道**和整个时间轴**池化，
#:   所以窗口内的共同偏移会被大部分归掉。
#:
#: 为什么要有这个开关：真机误报的主成分是 0.3–0.9 Hz 的慢漂移（步态频带只占
#: 7.9% 能量），而历史窗是 25 帧 = 500 ms —— 这种漂移在一个窗口内近似就是一个
#: 共同偏移。``"group"`` 是同事 Go2 归档件里的做法，对这个失效模式有直接动机。
NORMS = ("layer", "group")


def _make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "layer":
        return ChannelLayerNorm(channels)
    if kind == "group":
        if channels % 4:
            raise ValueError(f"group 归一化要求通道数是 4 的倍数，给的是 {channels}")
        return nn.GroupNorm(4, channels)
    raise ValueError(f"未知的归一化方式 {kind!r}，可选 {NORMS}")


class CausalBlock(nn.Module):
    """一层因果 TCN：左补零的 Conv1d + 归一化 + GELU。

    没有残差连接 —— 原件的 forward 是纯串联（`_5(_4(_3(_2(_1(_0(x))))))`），
    只有 4 层，不加也不会有梯度问题。
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 residual: bool = False, norm: str = "layer"):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = _make_norm(norm, channels)
        self.act = nn.GELU()
        #: 残差连接。参考实现没有（它也没有空洞），但**加了空洞之后残差就不是可选项**：
        #: 空洞卷积有 gridding 伪影 —— 从输出位置回溯，可达偏移是 a₁·1+a₂·2+a₃·4+a₄·8
        #: (aᵢ∈{0,1,2})，偏移 0（最新帧）只有一条通路，中间偏移有好几条。
        #: 实测最旧帧的敏感度是最新帧的 4 倍，而当前帧才是判断"我现在受多大力"
        #: 最直接的证据。残差给每一层一条直通路，把这个偏置补回来。
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.conv(nn.functional.pad(x, (self.pad, 0)))))
        return x + out if self.residual else out


class ForceTCN(nn.Module):
    """吃 (B, T, D) 的**已归一化**历史，吐 (values (B,15), confidence (B,4))。

    归一化不在这里做 —— 训练时在 DataLoader 那头做，导出时用
    :class:`NormalizedForceTCN` 把 mean/std 焊进图里。这样训练可以换归一化不用改结构。
    """

    def __init__(
        self,
        input_dim: int = spec.FRAME_SIZE,
        channels: int = 64,
        blocks: int = 4,
        kernel_size: int = 3,
        dilations: "list[int] | tuple[int, ...]" = (1, 2, 4, 8),
        residual: bool = False,
        norm: str = "layer",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = blocks
        self.residual = residual
        self.norm_kind = norm
        #: 每个 TCN 块的空洞率。**必须显式给**，没有 ``None`` 回退 ——
        #: 以前 ``None`` 悄悄表示"全 1"，结果一次残差消融因为漏传参数被静默改成
        #: 非空洞版，整轮实验作废。要跑非空洞对照就写 ``(1, 1, 1, 1)``。
        #:
        #: **为什么要用空洞**：全 1 时感受野只有 ``1 + 4×(3−1) = 9`` 帧 = 180 ms，
        #: 而 BPX 的步态周期是 0.45 s = 22.5 帧（`bpx_rma` 的 `GAIT_PERIOD`）——
        #: 网络连半个步态周期都看不到，没法把"支撑相在蹬地"和"有人在拉我"分开。
        #: 历史窗口有 25 帧（500 ms，整整一个周期），数据在那儿，卷积够不着而已。
        #:
        #: ``[1, 2, 4, 8]`` 下感受野 = ``1 + 2×(1+2+4+8) = 31`` 帧 = 620 ms，
        #: 覆盖一个完整周期还有余量。**参数量一点不变** —— 空洞是免费的。
        #:
        #: 实测（同数据同超参，测试集 22k 样本）：
        #:   xy 向量 RMSE 4.945 → 4.410 N（−10.8%）
        #:   xy 方向均值 14.57° → 12.26°（−15.9%），15–30 N 段 8.04° → 6.58°
        #: 所以默认就是 ``(1, 2, 4, 8)``。但注意**零力误报没有改善**
        #: （P95 5.11 → 5.06 N）—— 那一截误差不是感受野造成的，见
        #: `docs/traction_plan.md` 对零力误差下限的分析。
        if not dilations:
            raise ValueError("dilations 必须显式给出，非空洞对照请写 (1, 1, 1, 1)")
        self.dilations = [int(d) for d in dilations]
        if len(self.dilations) != blocks:
            raise ValueError(
                f"给了 {len(self.dilations)} 个空洞率，但有 {blocks} 个块"
            )
        if any(d < 1 for d in self.dilations):
            raise ValueError(f"空洞率必须 ≥1，给的是 {self.dilations}")

        self.encoder = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.tcn = nn.Sequential(
            *[CausalBlock(channels, kernel_size, d, residual, norm) for d in self.dilations]
        )
        self.values = nn.Linear(channels, spec.OUTPUT_SIZE)
        self.confidence = nn.Linear(channels, spec.CONFIDENCE_SIZE)

        # 激活表预解析成 (起, 止, 是否tanh, scale, offset)，前向里不再查字典
        self._heads: list[tuple[int, int, bool, float, float]] = []
        for (name, _), (begin, end) in zip(spec.OUTPUT_TERMS, spec.output_slices()):
            act, scale, offset = spec.OUTPUT_ACTIVATIONS[name]
            self._heads.append((begin, end, act == "tanh", scale, offset))

    def receptive_field(self) -> int:
        """能看到多少帧历史。编码器 kernel=1 不扩，每个 TCN 块扩 (k−1)·dilation。"""
        return 1 + sum((self.kernel_size - 1) * d for d in self.dilations)

    def receptive_field_s(self) -> float:
        """感受野换算成秒 —— 这同时是响应延迟的结构下限、和能看到多少步态周期。"""
        return self.receptive_field() * spec.SAMPLE_PERIOD_S

    def gait_cycles_visible(self, gait_period_s: float = 0.45) -> float:
        """感受野覆盖几个步态周期。<1 的话网络看不到周期性模式，
        分不清"支撑相在蹬地"和"有外力"。"""
        return self.receptive_field_s() / gait_period_s

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, D) -> (B, D, T)
        x = history.transpose(1, 2)
        x = self.tcn(self.encoder(x))
        last = x[:, :, -1]                      # (B, C) 只取最新帧
        raw = self.values(last)                 # (B, 15)

        parts = []
        for begin, end, is_tanh, scale, offset in self._heads:
            chunk = raw[:, begin:end]
            if is_tanh:
                parts.append(torch.tanh(chunk) * scale)
            else:
                parts.append(torch.sigmoid(chunk) * scale + offset)
        values = torch.cat(parts, dim=1)

        confidence = torch.sigmoid(self.confidence(last))   # (B, 4)
        return values, confidence


def from_checkpoint(ck: dict) -> "ForceTCN":
    """按检查点里记的结构造网络并装权重。

    **不要手搓 `ForceTCN(...)` 再 `load_state_dict`。** 残差连接不引入任何参数，
    `state_dict` 的键和形状完全一样，所以漏传 `residual` 时权重会**静默**装进一个
    结构不同的网络 —— 不报错，只是算出来的东西不对。2026-10-26 就是这么把一次
    真机选型搞反的。（空洞率以前也有同样的坑，`None` 悄悄表示全 1，现在已经
    改成必须显式给。）

    `norm` 不同则键名会变（LayerNorm 是 `norm.norm.*`，GroupNorm 是 `norm.*`），
    装错会直接报 missing/unexpected key —— 这一项不会静默失败，但还是记在检查点里。
    """
    model = ForceTCN(
        input_dim=ck["input_dim"],
        channels=ck["channels"],
        blocks=ck["blocks"],
        dilations=tuple(ck["dilations"]),
        residual=bool(ck.get("residual", False)),
        norm=str(ck.get("norm", "layer")),
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


class NormalizedForceTCN(nn.Module):
    """把 (x − mean) / std 焊进图里，导出 TorchScript / ONNX 用这个。

    板子那头只会拿到原始量纲的 61 维帧，归一化必须在图内部完成，否则一旦
    normalization.npz 和模型走散，错得无声无息。
    """

    def __init__(self, model: ForceTCN, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.reshape(1, 1, -1).float())
        self.register_buffer("std", std.reshape(1, 1, -1).float())

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model((history - self.mean) / self.std)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    net = ForceTCN()
    v, c = net(torch.zeros(2, spec.HISTORY_LENGTH, spec.FRAME_SIZE))
    print(f"参数        {count_parameters(net):,}")
    print(f"感受野      {net.receptive_field()} 帧 = {net.receptive_field_s() * 1000:.0f} ms")
    print(f"输出形状    values {tuple(v.shape)}  confidence {tuple(c.shape)}")
    fb, fe = spec.output_slice("force_b")
    print(f"力输出范围  ±{spec.OUTPUT_ACTIVATIONS['force_b'][1]:g} N   "
          f"零输入时 {v[0, fb:fe].tolist()}")
