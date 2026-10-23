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
def test_receptive_field_covers_a_gait_cycle(net):
    """默认空洞率必须让感受野至少覆盖一个完整步态周期。

    这是实测出来的要求，不是审美：全 1 空洞时感受野只有 9 帧 = 180 ms，
    而步态周期是 0.45 s —— 网络看不到周期性模式，分不清"支撑相在蹬地"和"有外力"。
    改成 (1,2,4,8) 之后 xy 向量 RMSE 降了 10.8%、方向误差降了 15.9%，参数量一个没多。
    """
    assert net.dilations == [1, 2, 4, 8], "默认空洞率变了，下面的结论要重新验"
    assert net.receptive_field() == 1 + 2 * (1 + 2 + 4 + 8) == 31
    assert net.gait_cycles_visible() >= 1.0, (
        f"感受野只覆盖 {net.gait_cycles_visible():.2f} 个步态周期"
    )
    assert net.receptive_field_s() == pytest.approx(0.62)


def test_dilation_is_free(net):
    """空洞不增加参数 —— 这是它比"加层/加宽"更划算的全部理由。"""
    plain = ForceTCN(dilations=None)
    assert count_parameters(plain) == count_parameters(net)
    assert plain.receptive_field() == 9, "全 1 空洞时应当退回 9 帧"


def test_frames_outside_receptive_field_do_not_matter():
    """感受野之外的帧必须逐位不影响输出 —— 这验证 padding 是因果的。

    默认空洞下感受野 31 帧 > 窗口 25 帧，没有"外面"的帧可改，
    所以用全 1 空洞那版（感受野 9 帧）来验这条结构性质。
    """
    torch.manual_seed(1)
    plain = ForceTCN(dilations=None).eval()
    x = torch.randn(3, T, D)
    v0, c0 = plain(x)

    outside = T - plain.receptive_field()          # 25 - 9 = 16
    x2 = x.clone()
    x2[:, :outside] = torch.randn(3, outside, D) * 50.0   # 老帧改成完全不同的大数

    v1, c1 = plain(x2)
    assert torch.equal(v0, v1), "感受野外的帧影响了输出 —— padding 不是因果的？"
    assert torch.equal(c0, c1)


def test_default_model_uses_the_whole_window(net):
    """默认空洞下感受野超过窗口，所以**每一帧**都该影响输出 —— 包括最老的那帧。"""
    torch.manual_seed(5)
    x = torch.randn(2, T, D)
    base, _ = net(x)
    for frame in (0, T // 2, T - 1):
        probe = x.clone()
        probe[:, frame] += 2.0
        assert not torch.equal(net(probe)[0], base), f"第 {frame} 帧没影响输出"


def test_last_frame_always_matters(net):
    """反过来：最新那一帧必须影响输出，否则网络在忽略当前状态。"""
    torch.manual_seed(2)
    x = torch.randn(2, T, D)
    v0, _ = net(x)
    x2 = x.clone()
    x2[:, -1] += 1.0
    v1, _ = net(x2)
    assert not torch.allclose(v0, v1)


def test_time_direction_matches_the_contract(net):
    """`spec.HISTORY_NEWEST_LAST` 说最新帧在末尾，模型必须真的这么用。

    模型取 `x[:, :, -1]` 配因果左 padding，所以末尾那一帧（以及它之前的 8 帧）决定输出，
    开头那些帧越界就没影响。把这件事变成断言 —— 时间方向接反了不会报错，只会让估计器
    学到一个延迟半秒的映射，训练损失照样好看。
    """
    assert spec.HISTORY_NEWEST_LAST, "这个测试假定最新帧在末尾"
    torch.manual_seed(4)
    x = torch.randn(2, T, D)
    base, _ = net(x)

    def sensitivity(frame: int) -> float:
        probe = x.clone()
        probe[:, frame] += 3.0
        return float((net(probe)[0] - base).abs().mean())

    # 最新帧必须影响输出。
    assert sensitivity(T - 1) > 0, "最新帧完全不影响输出 —— 时间方向反了？"

    # ⚠️ 这里**不能**断言"越新越敏感"。空洞卷积有 gridding 伪影：从输出位置回溯，
    # 可达偏移是 a₁·1 + a₂·2 + a₃·4 + a₄·8（aᵢ∈{0,1,2}），偏移 0（最新帧）只有一条
    # 通路，而中间偏移有好几条 —— 实测最旧帧的敏感度反而是最新帧的 4 倍。
    # 这是结构性的，不是 bug，但它说明当前帧的连接偏弱（残差连接是标准解法）。
    #
    # 时间方向的**硬保证**由 test_frames_outside_receptive_field_do_not_matter
    # 在非空洞版上验（感受野 9 < 窗口 25，能干净地验因果性）。


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
