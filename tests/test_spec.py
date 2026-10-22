"""契约自洽性：61 维分段、关节顺序、和 bpx_rma 的接口。

这些断言大部分在 `spec.py` 导入时就跑过一遍了。这里重复一遍是为了让它们出现在测试报告里，
另外补上几条跨模块的（spec ↔ bpx_rma ↔ normalize）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpx_rma import spec as rma_spec   # noqa: E402
from bpx_traction import spec          # noqa: E402
from bpx_traction.normalize import frame_columns  # noqa: E402


# --------------------------------------------------------------------------
# 输入帧
# --------------------------------------------------------------------------
def test_frame_size_is_61():
    assert spec.FRAME_SIZE == 61
    assert sum(dim for _, dim in spec.TERMS) == spec.FRAME_SIZE


def test_term_slices_are_contiguous_and_ordered():
    spans = spec.term_slices()
    assert spans[0][0] == 0
    assert spans[-1][1] == spec.FRAME_SIZE
    for (_, end), (begin, _) in zip(spans[:-1], spans[1:]):
        assert end == begin, "分段必须首尾相接，不能有缝也不能重叠"


def test_term_slice_lookup():
    assert spec.term_slice("imu_acc_b") == (0, 3)
    assert spec.term_slice("quat_wb") == (9, 13)
    assert spec.term_slice("joint_torque") == (37, 49)
    assert spec.term_slice("last_action") == (49, 61)
    with pytest.raises(KeyError):
        spec.term_slice("velocity_command")   # 已被物理删除，不该还能查到


def test_command_channel_is_absent():
    """指令通道必须**不在**估计器输入里 —— 这是断开闭环循环依赖的结构保证。"""
    assert all("command" not in name for name, _ in spec.TERMS)
    assert spec.FRAME_SIZE == 64 - 3, "61 = Go2 的 64 维契约减掉 3 维速度指令"


def test_last_action_is_present():
    """last_action 不能跟着指令一起删：估计器要靠它区分自主加速和外力。"""
    assert spec.term_slice("last_action")[1] - spec.term_slice("last_action")[0] == 12


# --------------------------------------------------------------------------
# 关节顺序
# --------------------------------------------------------------------------
def test_joint_order_follows_bpx_rma():
    """不在这里复制一份关节顺序，必须是 bpx_rma 的同一个对象。"""
    assert spec.JOINT_NAMES is rma_spec.JOINT_NAMES
    assert spec.NUM_JOINTS == 12


def test_per_joint_terms_are_12_wide():
    for term in spec.PER_JOINT_TERMS:
        begin, end = spec.term_slice(term)
        assert end - begin == spec.NUM_JOINTS, f"{term} 应该每关节一维"


def test_column_names_use_joint_names_not_indices():
    """按关节的维用关节名命名 —— 关节顺序一旦变了，归一化文件的列名会对不上而报错，
    不会悄悄错位。"""
    cols = frame_columns()
    begin, _ = spec.term_slice("joint_torque")
    assert cols[begin] == f"joint_torque[{rma_spec.JOINT_NAMES[0]}]"
    assert cols[begin + 11] == f"joint_torque[{rma_spec.JOINT_NAMES[11]}]"
    assert len(cols) == len(set(cols)) == spec.FRAME_SIZE


def test_quaternion_columns_match_declared_order():
    cols = frame_columns()
    begin, end = spec.term_slice("quat_wb")
    assert [c.split("[")[1].rstrip("]") for c in cols[begin:end]] == list(spec.QUAT_ORDER)
    assert spec.QUAT_ORDER == "wxyz"
    assert spec.QUAT_ORDER_ON_BOARD == "xyzw", (
        "板子的 projected_gravity_xyzw() 按 xyzw 解 getImuQuatHighRate()，"
        "板端适配器必须重排"
    )


# --------------------------------------------------------------------------
# 输出头
# --------------------------------------------------------------------------
def test_output_slices():
    assert spec.OUTPUT_SIZE == 15
    assert spec.CONFIDENCE_SIZE == 4
    assert spec.output_slice("force_b") == (12, 15)
    spans = spec.output_slices()
    assert spans[0][0] == 0 and spans[-1][1] == spec.OUTPUT_SIZE


def test_every_output_head_has_an_activation():
    assert set(spec.OUTPUT_ACTIVATIONS) == {name for name, _ in spec.OUTPUT_TERMS}
    for name, (act, scale, offset) in spec.OUTPUT_ACTIVATIONS.items():
        assert act in ("tanh", "sigmoid"), f"{name} 的激活 {act} 不认识"
        assert scale > 0


def test_height_head_covers_bpx_standing_height():
    """BPX 站 0.32~0.40 m（rewards.base_height.target_height），头的量程必须覆盖。
    Go2 那个头上界是 0.150，照搬会把高度估死在上界。"""
    act, scale, offset = spec.OUTPUT_ACTIVATIONS["base_height"]
    assert act == "sigmoid"
    assert offset <= 0.30 and offset + scale >= 0.42


def test_force_scale_matches_head():
    assert spec.OUTPUT_ACTIVATIONS["force_b"][1] == spec.FORCE_SCALE_N == 80.0


# --------------------------------------------------------------------------
# 时序
# --------------------------------------------------------------------------
def test_sample_period_matches_policy_rate():
    """估计器必须和策略同频，一个策略步推一帧历史。"""
    assert spec.SAMPLE_PERIOD_S == pytest.approx(1.0 / rma_spec.POLICY_HZ)
    assert spec.HISTORY_LENGTH == 25


# --------------------------------------------------------------------------
# 和策略帧的接口
# --------------------------------------------------------------------------
def test_command_slot_is_three_wide():
    begin, end = spec.policy_command_slice()
    assert end - begin == 3, "牵引层唯一的写入点必须是 3 维 (vx, vy, wz)"


def test_command_slot_read_dynamically_not_hardcoded():
    """策略帧会扩展（Sprint6 在末尾追加 [sinφ, cosφ] → 47 维），但新项一律追加在末尾，
    所以 cmd 槽的位置不变。这个测试保证我们是**读**出来的，不是写死的。"""
    begin, end = spec.policy_command_slice()
    names = [name for name, _ in rma_spec.OBS_TERMS]
    dims = [dim for _, dim in rma_spec.OBS_TERMS]
    idx = names.index("velocity_command")
    assert begin == sum(dims[:idx]) and end == begin + dims[idx]


def test_policy_frame_size_is_whatever_bpx_rma_says():
    assert spec.policy_frame_size() == rma_spec.OBS_FRAME_SIZE
    assert spec.policy_frame_size() in (45, 47), (
        f"策略帧是 {spec.policy_frame_size()} 维 —— 45（基线）或 47（相位进观测）之外的值"
        "说明 bpx_rma 那边动了帧结构，牵引层要跟着复核"
    )


# --------------------------------------------------------------------------
# 待确认的物理参数
# --------------------------------------------------------------------------
def test_tether_offset_range_is_well_formed():
    """挂点是个区域不是一个点（真机上人想从哪儿拉就从哪儿拉），范围本身得合法。"""
    ranges = spec.TETHER_OFFSET_RANGE_B
    assert len(ranges) == 3, "三个轴"
    for axis, (lo, hi) in zip("xyz", ranges):
        assert lo <= hi, f"{axis} 轴范围顺序反了：{(lo, hi)}"


def test_nominal_offset_is_inside_the_range():
    """标称挂点（确定性评估用）必须落在随机范围内，否则评估和训练分布脱节。"""
    for value, (lo, hi) in zip(spec.TETHER_OFFSET_B, spec.TETHER_OFFSET_RANGE_B):
        assert lo <= value <= hi


def test_tether_attach_point_is_flagged_unconfirmed():
    """挂点范围现在是按整机尺寸估的占位值。真机量过之后把 TETHER_OFFSET_CONFIRMED
    改 True，这个测试会提醒更新。"""
    if not spec.TETHER_OFFSET_CONFIRMED:
        pytest.xfail("挂点范围未经真机确认 —— 采数据前必须量（见 docs/traction_plan.md）")
    assert spec.TETHER_BODY in (rma_spec.TORSO_BODY,)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
