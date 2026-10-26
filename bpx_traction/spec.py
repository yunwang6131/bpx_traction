"""牵引层的接口契约。

这里的每个常数都有唯一来源，改之前先确认来源那边一致：

* **机器人规格 / 部署契约**：一律从 :mod:`bpx_rma.spec` 读，**不在这里复制一份**。
  关节顺序、q0、动作尺度、PD 增益、策略帧分段全是那边的权威值。
* **估计器 I/O**：`reference/core_code/contracts/estimator_io.yaml` +
  `reference/core_code/inference/run_estimator_rollouts.py` 的拼帧代码。
  两处逐项一致，见 `reference/PROVENANCE.md` §1.1。
* **BPX 专有的改动**（61 维而非 64、机身系而非世界系、height 头量程）：
  理由写在各自的常数旁边，完整论证见 `docs/traction_plan.md`。

本模块是**纯 Python**，不 import torch / numpy / isaaclab，所以板子端的轻量实现和
离线契约检查都能用。
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent          # .../bpx/bpx_traction
PROJECT_DIR = PACKAGE_DIR.parent                       # .../bpx
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from bpx_rma import spec as rma_spec  # noqa: E402
except ImportError:
    # 板子上没有 gymnasium，而 `bpx_rma/__init__.py` 第一行就 import 它。
    # `bpx_rma/spec.py` 本身是纯 Python（只用 os / pathlib），绕开包的 __init__
    # 直接按文件加载 —— 这样板端和开发机读的是同一份契约，不会出现两套常数。
    import importlib.util as _ilu

    _p = PROJECT_DIR / "bpx_rma" / "spec.py"
    if not _p.is_file():
        raise
    _s = _ilu.spec_from_file_location("bpx_rma_spec_standalone", _p)
    rma_spec = _ilu.module_from_spec(_s)
    _s.loader.exec_module(rma_spec)

REFERENCE_DIR = PACKAGE_DIR / "reference"


# ---------------------------------------------------------------------------
# 时序：和策略同频，一个策略步正好一个估计器步
# ---------------------------------------------------------------------------
#: 估计器的采样周期。契约里是 `sample_period_s: 0.02`。
SAMPLE_PERIOD_S = 0.02
#: 历史窗口长度。契约里是 `history_steps: 25`。
HISTORY_LENGTH = 25

#: 历史窗口的时间方向：**最旧在前，最新在末尾**（`history[-1]` 是当前帧）。
#:
#: 这条必须写死，因为三处约定必须对齐，而参考实现用的是**反过来的**：
#:
#:   * 板子：`history[:-1] = history[1:]; history[-1] = frame` —— 最新在末尾；
#:   * `model.py`：卷积左 padding（因果），前向取 `x[:, :, -1]` —— 最新必须在末尾；
#:   * Go2 参考实现：`torch.roll(history, 1, 1); history[:, 0] = sensor` —— 最新在**开头**。
#:
#: 时间方向接反了不会报错，只会让估计器把"最早的 9 帧"当成当前状态去推外力 ——
#: 训练损失照样下降（它学到的是一个延迟 500 ms 的映射），上真机才发现响应完全不对。
HISTORY_NEWEST_LAST = True

assert abs(SAMPLE_PERIOD_S - 1.0 / rma_spec.POLICY_HZ) < 1e-12, (
    "估计器周期和策略频率对不上：估计器要和策略同频，一个策略步推一帧历史"
)

# `bpx_rma.spec.HISTORY_LENGTH` 恰好也是 25（板子的 adaptation 历史）。两者是**独立**的
# 契约，只是数值相同。好处是板子上一个环形缓冲的推进节拍能共用，坏处是容易误以为改一个
# 另一个会跟着变 —— 不会。这条断言在它们分家时提醒一下，不是要求它们永远相等。
_HISTORY_COINCIDES_WITH_POLICY = HISTORY_LENGTH == rma_spec.HISTORY_LENGTH


# ---------------------------------------------------------------------------
# 输入帧：61 维
# ---------------------------------------------------------------------------
# Go2 的契约是 64 维，末尾带 3 维速度指令。**BPX 把那 3 维物理删掉**，理由：
#
#   指令由牵引层自己产生（F̂ → 映射 → cmd），再喂回估计器就形成了闭环循环依赖。
#   Go2 的采集脚本是靠"把指令通道置零"来断开的（`collect_sim_estimator_dataset.py`
#   里 `proprio[:, 13:16] = 0.0`，注释写明 "so the estimator cannot see its own
#   generated command"），但那是个靠人守的约定 —— 同一个包里的
#   `run_estimator_rollouts.py` 就没置零，喂的是真实指令。
#   删掉是结构保证，置零是口头保证。
#
# `last_action` 必须留着：估计器要靠它区分"我被命令加速"和"有人在拉我"。
# 删了它，每一次自主加速都会被当成外力。
TERMS: list[tuple[str, int]] = [
    ("imu_acc_b", 3),            # IMU 线加速度（机身系，**含重力**）  m/s²
    ("imu_gyro_b", 3),           # 角速度（机身系）                    rad/s
    ("projected_gravity_b", 3),  # 投影重力，由四元数算                 1
    ("quat_wb", 4),              # 机身姿态四元数，见 QUAT_ORDER        1
    ("joint_pos_rel", 12),       # q − q0，SDK 顺序                     rad
    ("joint_vel", 12),           # dq，SDK 顺序                        rad/s
    ("joint_torque", 12),        # 关节力矩，SDK 顺序                   N·m
    ("last_action", 12),         # 上一步 raw action（已 clip ±4）      1
]
FRAME_SIZE = sum(dim for _, dim in TERMS)      # 61
HISTORY_INPUT_SHAPE = (HISTORY_LENGTH, FRAME_SIZE)   # (25, 61)

assert FRAME_SIZE == 61, f"输入帧应为 61 维，当前 {FRAME_SIZE}"

#: 按关节的三项必须和 `bpx_rma.spec.JOINT_NAMES`（SDK 顺序）逐位对齐。
#: Go2 的帧里混了两种关节顺序（位置/速度/动作是 leg-major，力矩是 type-major，
#: 见 PROVENANCE §2.5），BPX 一律用 SDK 顺序，由 `check_traction_contract.py` 断言。
PER_JOINT_TERMS = ("joint_pos_rel", "joint_vel", "joint_torque", "last_action")

#: 四元数顺序。契约 `frame_contract.yaml` 写的是 `order: wxyz, meaning: body_to_world`，
#: IsaacLab 的 `root_quat_w` 也是 wxyz，所以估计器的帧统一用 **wxyz**。
#:
#: ⚠️ 板子**不是** wxyz：`bpx_dual_pad.py` 的 `projected_gravity_xyzw()` 按 xyzw 解
#: `getImuQuatHighRate()`，而手册 6.3 给低频的 `getImuQuat` 标的却是 (w,x,y,z)、
#: 8.5 的高频版没标顺序。板子上策略能跑（投影重力错了会立刻摔）说明高频版实际是 xyzw，
#: 但这一条**没有文档支撑，且门 2 不覆盖它**（门 2 管关节方向/零位/顺序）。
#: 板子端适配器必须做 xyzw → wxyz 的重排，真机验证列在 `docs/traction_plan.md` §7.1。
QUAT_ORDER = "wxyz"
QUAT_ORDER_ON_BOARD = "xyzw"


def term_slices() -> list[tuple[int, int]]:
    """单帧内每个分段的 [起, 止)。"""
    out, begin = [], 0
    for _, dim in TERMS:
        out.append((begin, begin + dim))
        begin += dim
    return out


def term_slice(name: str) -> tuple[int, int]:
    """按名字取分段，例如 ``term_slice("joint_torque") -> (37, 49)``。"""
    for (term, _), span in zip(TERMS, term_slices()):
        if term == name:
            return span
    raise KeyError(f"输入帧里没有这一项：{name}（有的是 {[t for t, _ in TERMS]}）")


assert term_slices() == [
    (0, 3), (3, 6), (6, 9), (9, 13), (13, 25), (25, 37), (37, 49), (49, 61)
], "输入帧分段算错了"


# ---------------------------------------------------------------------------
# 输出头：15 维状态 + 4 维置信度
# ---------------------------------------------------------------------------
# 结构沿用 Go2（从已删权重包的 TorchScript 还原，见 PROVENANCE §2.1/§2.2），
# 但 height 头的量程改成 BPX 自己的。
#
# **只有 force 头有消费端**（BPX 的 45/47 维策略帧放不下其余四组）。四个辅助头仍然训，
# 它们是共享主干的正则项 —— Go2 给 height 的损失权重高达 8.0，删掉会直接改变主干学到的
# 表示。推理时不取而已。
OUTPUT_TERMS: list[tuple[str, int]] = [
    ("base_lin_vel_b", 3),   # 辅助：机身线速度（机身系）
    ("base_height", 1),      # 辅助：机身高度
    ("foot_contact", 4),     # 辅助：四足触地概率
    ("foot_load", 4),        # 辅助：四足负载（按体重归一）
    ("force_b", 3),          # ★ 消费：外力（机身系）
]
OUTPUT_SIZE = sum(dim for _, dim in OUTPUT_TERMS)    # 15

CONFIDENCE_TERMS: list[str] = ["vel", "height", "contact", "force"]
CONFIDENCE_SIZE = len(CONFIDENCE_TERMS)              # 4

assert OUTPUT_SIZE == 15 and CONFIDENCE_SIZE == 4


def output_slices() -> list[tuple[int, int]]:
    out, begin = [], 0
    for _, dim in OUTPUT_TERMS:
        out.append((begin, begin + dim))
        begin += dim
    return out


def output_slice(name: str) -> tuple[int, int]:
    for (term, _), span in zip(OUTPUT_TERMS, output_slices()):
        if term == name:
            return span
    raise KeyError(f"输出里没有这一项：{name}（有的是 {[t for t, _ in OUTPUT_TERMS]}）")


assert output_slices() == [(0, 3), (3, 4), (4, 8), (8, 12), (12, 15)]

#: 各输出头的激活与量程。`tanh` 头的值域是 ±scale，`sigmoid` 头是 [offset, offset+scale]。
OUTPUT_ACTIVATIONS: dict[str, tuple[str, float, float]] = {
    #  名字            激活        scale  offset
    "base_lin_vel_b": ("tanh",     2.0,   0.0),
    "base_height":    ("sigmoid",  0.3,   0.2),   # → 0.20 ~ 0.50 m
    "foot_contact":   ("sigmoid",  1.0,   0.0),
    "foot_load":      ("sigmoid",  2.0,   0.0),
    "force_b":        ("tanh",    80.0,   0.0),   # → ±80 N
}
assert set(OUTPUT_ACTIVATIONS) == {name for name, _ in OUTPUT_TERMS}

# height 头量程的依据：BPX 的 `rewards.base_height.target_height` 在 0.32（fast）~
# 0.40（flat）之间，sim2sim 实测站立 0.381、1.5 m/s 时 0.370、3.0 m/s 时 0.318。
# 0.20~0.50 覆盖全程且留了余量。Go2 那个头的上界是 0.150，**不能照搬**。
_HEIGHT_LO, _HEIGHT_HI = 0.20, 0.50
assert OUTPUT_ACTIVATIONS["base_height"][2] == _HEIGHT_LO
assert OUTPUT_ACTIVATIONS["base_height"][1] + _HEIGHT_LO == _HEIGHT_HI


# ---------------------------------------------------------------------------
# 力的量纲
# ---------------------------------------------------------------------------
#: 力的归一化尺度。`baseline_manifest.yaml` 的 `force_scale_n: 80`，也是 force 头的
#: tanh 量程。Go2 整机约 15 kg、BPX-Pro 15.1 kg（手册 9.1），量级可直接沿用。
FORCE_SCALE_N = 80.0

#: "有力"的判定阈值。损失里 ‖F‖ > 2 N 的样本加权，验收里方向误差只在 ≥2 N 时才有定义
#: （零力的方向是未定义的，不是"角度算错了"）。
FORCE_ACTIVE_N = 2.0

#: 设计力范围。**上限 30 N 是扛力扫描实测出来的，不是估的**
#: （`logs/traction/drag_sweep.json`，repeats=4，三个策略 × 8 方向 × 3 指令）：
#:
#:   * rollfix / sprint7 在 ≤30 N 的全部方向和指令下**零失败**，drag ≤ 0.28 m/s；
#:   * 到 40 N 两者都有 4–7 个方向摔倒，drag 跳到 0.24–0.76 m/s；
#:   * flat 明显更弱 —— 站立 10 N 就开始摔，不能做牵引基底。
#:
#: 力超过 30 N 时策略压不住外力，机器人被**直接拖着动**，"发什么速度就走什么速度"
#: 这个前提不再成立，映射发什么都没意义。所以 30 N 是这条链路的物理上限，不是偏好。
#: 手册 9.1 的最大有效载荷 8 kg ≈ 78 N 是**结构**上限，比这个宽得多，不是约束。
FORCE_DESIGN_RANGE_N = (0.0, 30.0)
FORCE_OOD_RANGE_N = (40.0, 55.0)

assert OUTPUT_ACTIVATIONS["force_b"][1] == FORCE_SCALE_N


# ---------------------------------------------------------------------------
# 坐标系
# ---------------------------------------------------------------------------
#: 力的输出坐标系。**BPX 用机身系，不用世界系。**
#:
#: Go2 的 `frame_contract.yaml` 把世界系 W0（启动朝向 + 重力对齐）定为规范量，机身系是
#: 派生量。BPX 反过来，理由三条：
#:   1. 消费端（速度指令）本来就是机身系，省掉一次往返；
#:   2. 世界系要跟踪绝对 yaw，而 BPX 的 IMU yaw 会漂（Go2 的鲁棒测试专门注入
#:      3.0°/场景的 yaw 漂移来测这个脆弱点）；
#:   3. 牵引只发 (vx, vy, 0)，机器人不主动转向，世界系带来的"方向保持"收益用不上。
#: 如果长场景里方向保持不住，再切 W0 —— 这是可回退的决定。
FORCE_FRAME = "body"

#: IMU 与身体坐标系的关系。手册 2.3.3：「IMU 传感器固连在身体内部，其坐标系与身体坐标系
#: 平行（无旋转变换）」。所以加速度计的轴就是机身轴，不需要标安装旋转。
IMU_ALIGNED_WITH_BODY = True


# ---------------------------------------------------------------------------
# 挂点：绳子系在机器人哪里
# ---------------------------------------------------------------------------
#: 牵引绳的挂点所在刚体。
TETHER_BODY = rma_spec.TORSO_BODY

#: 标称挂点相对该刚体原点的偏移（机身系，米）。确定性评估（扛力扫描、回放）用这个。
#:
#: 参考尺寸（手册 9.1）：站立 675 × 356 × 425 mm。
TETHER_OFFSET_B: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: **挂点不是一个固定死点，是一个区域。** 采训练数据时按 env 在这个范围里随机。
#:
#: 真机上人想从哪儿拉就从哪儿拉 —— 前胸、后背、把手，位置不固定。所以估计器必须对
#: 挂点**不变**，而做到不变的办法是训练时随机化，不是把它钉死。
#:
#: 代价要明说：同一个力作用在不同挂点上，产生的本体响应不一样（力臂不同 → 俯仰/横滚
#: 力矩不同）。随机化之后，"本体感知 → 外力"这个映射就是**一对多**的，估计器只能学到
#: 条件均值，这部分误差是不可消除的。挂点范围越大，这个下限越高。
#:
#: 所以它应当是个**可测量的消融项**，不是一个想当然的设定：固定挂点训一版、随机挂点训
#: 一版，比力精度的差，就是"挂哪儿都行"这个便利的标价。见 `docs/traction_plan.md`。
#:
#: ⚠️ 当前范围是按整机尺寸估的占位值，要照真机实际能挂的位置量一遍。
TETHER_OFFSET_RANGE_B: tuple[tuple[float, float], ...] = (
    (-0.15, 0.15),    # x 前后：躯干长 0.675 m，取中段
    (-0.05, 0.05),    # y 左右：贴近中线
    (0.00, 0.10),     # z 上下：机身上方（背带/把手）
)
TETHER_OFFSET_CONFIRMED = False   # 真机量过挂点范围并更新上面的值之后改 True


# ---------------------------------------------------------------------------
# 和策略帧的接口：只写 velocity_command 这一个槽
# ---------------------------------------------------------------------------
def policy_command_slice() -> tuple[int, int]:
    """策略帧里 `velocity_command` 的 [起, 止)。

    **从 `bpx_rma.spec` 动态读，不写死。** 策略帧会扩展（Sprint6 在 45 维末尾追加了
    `[sinφ, cosφ]` 变成 47 维），但新项一律追加在末尾、前 45 维一字不动，所以 cmd 槽
    的位置不变。这个函数保证帧再怎么加项，牵引层都指到正确的位置。
    """
    for (name, _), span in zip(rma_spec.OBS_TERMS, rma_spec.obs_term_slices()):
        if name == "velocity_command":
            return span
    raise KeyError(
        "策略帧里找不到 velocity_command —— 牵引层唯一的写入点没了，"
        f"当前帧是 {[n for n, _ in rma_spec.OBS_TERMS]}"
    )


def policy_frame_size() -> int:
    """当前策略帧的维数（45 或 47，取决于 bpx_rma 那边的配置）。"""
    return rma_spec.OBS_FRAME_SIZE


# 牵引层不依赖策略帧的总维数，只依赖 cmd 槽存在且是 3 维。
_cmd_begin, _cmd_end = policy_command_slice()
assert _cmd_end - _cmd_begin == 3, "velocity_command 必须是 3 维 (vx, vy, wz)"


# ---------------------------------------------------------------------------
# 从 bpx_rma 透传的量（方便引用，不复制值）
# ---------------------------------------------------------------------------
JOINT_NAMES = rma_spec.JOINT_NAMES          # SDK 顺序，12 个
NUM_JOINTS = rma_spec.NUM_JOINTS            # 12
DEFAULT_JOINT_POS = rma_spec.DEFAULT_JOINT_POS
ACTION_SCALE = rma_spec.ACTION_SCALE
ACTION_CLIP = rma_spec.ACTION_CLIP
POLICY_HZ = rma_spec.POLICY_HZ

for _term in PER_JOINT_TERMS:
    assert term_slice(_term)[1] - term_slice(_term)[0] == NUM_JOINTS, (
        f"{_term} 应该是 {NUM_JOINTS} 维（每关节一个）"
    )


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
#: 数据集、模型、报告都落这里。`logs/` 在 .gitignore 里，不进版本库。
TRACTION_LOG_DIR = rma_spec.LOG_DIR / "traction"

#: 当前在用的力估计器。所有脚本的 `--estimator/--model` 默认值都从这里取，
#: 换模型只改这一行。
#:
#: `estimator_real_ft` = `estimator_dilated` 用真机无外力录制做过域适配
#: （`scripts/finetune_real.py`：仿真批次全监督 + 真机批次 力→0 / 速度→低通指令）。
#:
#:     模型                      仿真 xy RMSE   仿真高力段   真机误报 p99
#:     estimator_dilated            4.535        7.452       48.00 N
#:     estimator_residual           4.615          —         更差，见 git log
#:     estimator_real_ft            4.631        7.829       35.55 N  <- 留出法实测
#:
#: 真机那一列是**时间留出法**量的（前 70% 训、后 30% 只评估，产物是
#: `estimator_real_ft_holdout`）。同一份数据不做留出会报 18.96 N —— 偏乐观一倍，
#: 别拿那个数做判断。仿真高力段只涨 5%，说明没退化成"永远输出零"。
#:
#: **仍然不能上真机**：留出法下要零误起步得把死区放到 30 N，正好等于设计范围上限
#: （`FORCE_DESIGN_RANGE_N`），可用区间是零。瓶颈是真机数据量 —— 样本外指标在
#: epoch 32 就平了，继续训没用。详见 `mapping.py` 的 `deadzone_on_n`。
DEFAULT_ESTIMATOR_DIR = TRACTION_LOG_DIR / "estimator_real_ft"


def summary() -> str:
    """一行行列出契约，给 `check_traction_contract.py` 和日志用。"""
    lines = [
        f"估计器输入   {HISTORY_LENGTH} × {FRAME_SIZE}  周期 {SAMPLE_PERIOD_S * 1000:.0f} ms",
    ]
    for (name, dim), (b, e) in zip(TERMS, term_slices()):
        lines.append(f"    [{b:2d}:{e:2d}]  {name:<22s} {dim:2d}")
    lines.append(f"估计器输出   {OUTPUT_SIZE} 状态 + {CONFIDENCE_SIZE} 置信度")
    for (name, dim), (b, e) in zip(OUTPUT_TERMS, output_slices()):
        act, scale, offset = OUTPUT_ACTIVATIONS[name]
        rng = f"±{scale:g}" if act == "tanh" else f"{offset:g}~{offset + scale:g}"
        star = " ★消费" if name == "force_b" else ""
        lines.append(f"    [{b:2d}:{e:2d}]  {name:<18s} {dim:2d}  {act:<8s} {rng}{star}")
    lines.append(f"力           机身系，±{FORCE_SCALE_N:g} N，设计范围 {FORCE_DESIGN_RANGE_N} N")
    lines.append(f"策略帧       {policy_frame_size()} 维，cmd 槽 {policy_command_slice()}")
    lines.append(f"挂点         {TETHER_BODY} + {TETHER_OFFSET_B}"
                 f"{'' if TETHER_OFFSET_CONFIRMED else '   ⚠️ 未经真机确认'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
