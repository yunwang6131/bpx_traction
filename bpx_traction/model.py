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

2. **逐帧 LayerNorm。** 在通道维上归一化，不跨时间。跨时间的归一化（如 GroupNorm(1, C)）
   会让第 t 帧的输出依赖窗口内更晚的帧 —— 因为窗口固定所以不会出错，但那就不是因果网络了。

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


class CausalBlock(nn.Module):
    """一层因果 TCN：左补零的 Conv1d + 逐帧 LayerNorm + GELU。

    没有残差连接 —— 原件的 forward 是纯串联（`_5(_4(_3(_2(_1(_0(x))))))`），
    只有 4 层，不加也不会有梯度问题。
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = ChannelLayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))   # 只补左边 = 因果
        return self.act(self.norm(self.conv(x)))


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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = blocks

        self.encoder = nn.Conv1d(input_dim, channels, kernel_size=1)
        self.tcn = nn.Sequential(
            *[CausalBlock(channels, kernel_size) for _ in range(blocks)]
        )
        self.values = nn.Linear(channels, spec.OUTPUT_SIZE)
        self.confidence = nn.Linear(channels, spec.CONFIDENCE_SIZE)

        # 激活表预解析成 (起, 止, 是否tanh, scale, offset)，前向里不再查字典
        self._heads: list[tuple[int, int, bool, float, float]] = []
        for (name, _), (begin, end) in zip(spec.OUTPUT_TERMS, spec.output_slices()):
            act, scale, offset = spec.OUTPUT_ACTIVATIONS[name]
            self._heads.append((begin, end, act == "tanh", scale, offset))

    def receptive_field(self) -> int:
        """能看到多少帧历史。编码器 kernel=1 不扩，每个 TCN 块扩 (k−1)。"""
        return 1 + self.num_blocks * (self.kernel_size - 1)

    def receptive_field_s(self) -> float:
        """感受野换算成秒 —— 这是响应延迟的结构下限。"""
        return self.receptive_field() * spec.SAMPLE_PERIOD_S

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
