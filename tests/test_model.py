"""估计器网络：形状、因果性、输出量程、导出一致性。

最要紧的是**因果性**那两条。模型前向只取最后一帧（`x[:, :, -1]`），如果卷积用 same
padding，最后一帧会卷到右侧补的零 —— 训练和部署都"看到未来是零"，所以不会崩，
但感受野就不再是 9 帧了，响应延迟的分析也跟着失效。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_traction import spec                                        # noqa: E402
from bpx_traction.model import (                                      # noqa: E402
    ForceTCN,
    NormalizedForceTCN,
    count_parameters,
)

T, D = spec.HISTORY_LENGTH, spec.FRAME_SIZE


@pytest.fixture
def net():
    torch.manual_seed(0)
    return ForceTCN().eval()


# --------------------------------------------------------------------------
# 形状
# --------------------------------------------------------------------------
def test_output_shapes(net):
    values, confidence = net(torch.zeros(4, T, D))
    assert values.shape == (4, spec.OUTPUT_SIZE)
    assert confidence.shape == (4, spec.CONFIDENCE_SIZE)


def test_accepts_any_batch_size(net):
    for batch in (1, 7, 64):
        v, c = net(torch.randn(batch, T, D))
        assert v.shape[0] == c.shape[0] == batch


def test_parameter_count_is_small_enough_for_the_board(net):
    """55k 参数、CPU 上一次推理要塞进 20 ms 的策略步里。数量级跑偏要立刻知道。"""
    n = count_parameters(net)
    assert 40_000 < n < 80_000, f"参数量 {n:,} 偏离预期量级"


# --------------------------------------------------------------------------
# 因果性 —— 最重要的两条
# --------------------------------------------------------------------------
def test_receptive_field_is_nine_frames(net):
    assert net.receptive_field() == 1 + 4 * (3 - 1) == 9
    assert net.receptive_field_s() == pytest.approx(9 * spec.SAMPLE_PERIOD_S)
    assert net.receptive_field_s() == pytest.approx(0.18), "180 ms，响应延迟的结构下限"


def test_frames_outside_receptive_field_do_not_matter(net):
    """感受野是 9 帧，窗口是 25 帧 —— 改动最老的 16 帧，输出应当逐位不变。

    这同时验证了两件事：padding 是因果的（否则感受野会更大或含义不明），
    以及感受野的计算没算错。
    """
    torch.manual_seed(1)
    x = torch.randn(3, T, D)
    v0, c0 = net(x)

    outside = T - net.receptive_field()          # 25 - 9 = 16
    x2 = x.clone()
    x2[:, :outside] = torch.randn(3, outside, D) * 50.0   # 老帧改成完全不同的大数

    v1, c1 = net(x2)
    assert torch.equal(v0, v1), "感受野外的帧影响了输出 —— padding 不是因果的？"
    assert torch.equal(c0, c1)


def test_last_frame_always_matters(net):
    """反过来：最新那一帧必须影响输出，否则网络在忽略当前状态。"""
    torch.manual_seed(2)
    x = torch.randn(2, T, D)
    v0, _ = net(x)
    x2 = x.clone()
    x2[:, -1] += 1.0
    v1, _ = net(x2)
    assert not torch.allclose(v0, v1)


# --------------------------------------------------------------------------
# 输出量程 —— 激活必须按契约钳住
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.0, 1.0, -1.0, 100.0, -100.0])
def test_outputs_stay_in_contract_range(net, scale):
    """喂极端输入，每个头都必须落在 spec 声明的区间内。"""
    values, confidence = net(torch.full((2, T, D), float(scale)))

    for (name, _), (begin, end) in zip(spec.OUTPUT_TERMS, spec.output_slices()):
        act, head_scale, offset = spec.OUTPUT_ACTIVATIONS[name]
        chunk = values[:, begin:end]
        lo, hi = (-head_scale, head_scale) if act == "tanh" else (offset, offset + head_scale)
        assert chunk.min() >= lo - 1e-5, f"{name} 下越界：{chunk.min():.4f} < {lo}"
        assert chunk.max() <= hi + 1e-5, f"{name} 上越界：{chunk.max():.4f} > {hi}"

    assert confidence.min() >= 0.0 and confidence.max() <= 1.0


def test_force_head_saturates_at_80n(net):
    """力头是 tanh × 80，再大的激励也不该超过 ±80 N。"""
    begin, end = spec.output_slice("force_b")
    values, _ = net(torch.full((1, T, D), 1e4))
    assert values[:, begin:end].abs().max() <= spec.FORCE_SCALE_N + 1e-4


def test_height_head_in_bpx_range(net):
    begin, end = spec.output_slice("base_height")
    values, _ = net(torch.randn(16, T, D) * 10)
    assert values[:, begin:end].min() >= 0.20 - 1e-5
    assert values[:, begin:end].max() <= 0.50 + 1e-5


# --------------------------------------------------------------------------
# 导出
# --------------------------------------------------------------------------
def test_normalized_wrapper_equals_manual_normalisation(net):
    torch.manual_seed(3)
    mean, std = torch.randn(D), torch.rand(D) + 0.5
    wrapped = NormalizedForceTCN(net, mean, std).eval()

    x = torch.randn(5, T, D)
    with torch.no_grad():
        a_v, a_c = wrapped(x)
        b_v, b_c = net((x - mean) / std)
    assert torch.allclose(a_v, b_v, atol=1e-6)
    assert torch.allclose(a_c, b_c, atol=1e-6)


def test_torchscript_trace_matches_eager(net):
    """导出用的 trace 必须和 eager 一致，否则板子上跑的和训练的不是一个东西。"""
    mean, std = torch.zeros(D), torch.ones(D)
    wrapped = NormalizedForceTCN(net, mean, std).eval()
    example = torch.zeros(1, T, D)
    traced = torch.jit.trace(wrapped, example)

    x = torch.randn(8, T, D)
    with torch.no_grad():
        ev, ec = wrapped(x)
        tv, tc = traced(x)
    assert torch.allclose(ev, tv, atol=1e-6)
    assert torch.allclose(ec, tc, atol=1e-6)


def test_deterministic_in_eval_mode(net):
    """eval 下同样输入必须给同样输出（没有 dropout / BatchNorm 之类的状态）。"""
    x = torch.randn(4, T, D)
    with torch.no_grad():
        a = net(x)[0]
        b = net(x)[0]
    assert torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
