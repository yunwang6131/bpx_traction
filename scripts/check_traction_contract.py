#!/usr/bin/env python3
"""牵引层的契约闸门。改完代码、或者从 main 合并之后，跑这个。

查的是 **git 抓不到的东西**。牵引层只新增文件、不碰 `bpx_rma/`，所以合并永远不冲突 ——
但"不冲突"不等于"没问题"。底层那边改了命令区间、改了关节顺序、改了策略帧的分段，
文件层面毫无冲突，数却全错了。这个脚本就是抓这类静默失效的。

分两档跑：

    # 默认：纯 CPU，不启 Isaac Sim，和训练并行跑也不抢资源
    python scripts/check_traction_contract.py

    # 全量：额外启 Isaac Sim，核对映射限幅 == 当前任务的命令区间
    conda activate bpx_rma
    python scripts/check_traction_contract.py --with-env --headless

CI / 提交前用默认档就够。**改过 `bpx_rma/velocity_env_cfg.py` 的命令区间之后必须跑全量档。**
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ---------------------------------------------------------------------------
# 结果收集
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []   # (状态, 标题, 说明)

    def ok(self, title: str, detail: str = "") -> None:
        self.rows.append(("通过", title, detail))

    def warn(self, title: str, detail: str = "") -> None:
        self.rows.append(("注意", title, detail))

    def fail(self, title: str, detail: str = "") -> None:
        self.rows.append(("失败", title, detail))

    def skip(self, title: str, detail: str = "") -> None:
        self.rows.append(("跳过", title, detail))

    @property
    def failed(self) -> bool:
        return any(status == "失败" for status, _, _ in self.rows)

    def render(self) -> str:
        mark = {"通过": "  ok  ", "注意": " 注意 ", "失败": " 失败 ", "跳过": " 跳过 "}
        lines = []
        for status, title, detail in self.rows:
            lines.append(f"[{mark[status]}] {title}")
            if detail:
                for line in detail.splitlines():
                    lines.append(f"           {line}")
        counts = {s: sum(1 for st, _, _ in self.rows if st == s) for s in ("通过", "注意", "失败", "跳过")}
        lines.append("")
        lines.append(
            f"通过 {counts['通过']}   注意 {counts['注意']}   "
            f"失败 {counts['失败']}   跳过 {counts['跳过']}"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 闸门一：bpx_rma 零改动
# ---------------------------------------------------------------------------
def check_bpx_rma_untouched(report: Report) -> None:
    """牵引层的根本约定：`bpx_rma/` 一个字节都不改。

    这不是洁癖 —— 底层那条线（辨识验证）还在推进，四条对照策略全建在现有契约上。
    牵引一旦动了 `bpx_rma/`，两条线就缠在一起了：合并会冲突、对照实验会被污染、
    出了问题分不清是谁的。
    """
    try:
        merge_base = subprocess.run(
            ["git", "merge-base", "HEAD", "main"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--stat", merge_base, "--", "bpx_rma/"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        report.warn("bpx_rma 零改动", f"git 查不了：{exc}")
        return

    if diff:
        report.fail(
            "bpx_rma 零改动",
            "牵引分支动了底层文件：\n" + diff +
            "\n把改动挪进 bpx_traction/，用 cfg 组合而不是改 bpx_rma/",
        )
    else:
        report.ok("bpx_rma 零改动", f"相对 {merge_base[:8]} 无改动")


# ---------------------------------------------------------------------------
# 闸门二：契约自洽
# ---------------------------------------------------------------------------
def check_spec_self_consistent(report: Report) -> None:
    from bpx_traction import spec            # 导入时 spec.py 自己的断言就会跑

    spans = spec.term_slices()
    contiguous = spans[0][0] == 0 and spans[-1][1] == spec.FRAME_SIZE and all(
        a[1] == b[0] for a, b in zip(spans[:-1], spans[1:])
    )
    if not contiguous:
        report.fail("输入帧分段", f"分段不连续：{spans}")
        return
    report.ok("输入帧分段", f"{spec.HISTORY_LENGTH} × {spec.FRAME_SIZE}，"
                            f"周期 {spec.SAMPLE_PERIOD_S * 1000:.0f} ms")

    if any("command" in name for name, _ in spec.TERMS):
        report.fail("指令通道已删除", "估计器输入里出现了速度指令 —— 闭环循环依赖又接上了")
    else:
        report.ok("指令通道已删除", "61 = 64 − 3，结构上看不到自己产生的指令")

    if spec.term_slice("last_action")[1] - spec.term_slice("last_action")[0] != 12:
        report.fail("last_action 保留", "删了它，自主加速会被当成外力")
    else:
        report.ok("last_action 保留", "12 维，用于区分自主加速和外力")


def check_joint_order(report: Report) -> None:
    from bpx_rma import spec as rma_spec
    from bpx_traction import spec
    from bpx_traction.normalize import frame_columns

    if spec.JOINT_NAMES is not rma_spec.JOINT_NAMES:
        report.fail("关节顺序", "牵引层复制了一份关节顺序，必须直接引用 bpx_rma.spec")
        return

    cols = frame_columns()
    bad = []
    for term in spec.PER_JOINT_TERMS:
        begin, end = spec.term_slice(term)
        expect = [f"{term}[{j}]" for j in rma_spec.JOINT_NAMES]
        if cols[begin:end] != expect:
            bad.append(term)
    if bad:
        report.fail("关节顺序", f"这些项的列名和 SDK 顺序对不上：{bad}")
    else:
        report.ok("关节顺序", f"四项按关节的量全部 SDK 顺序（{rma_spec.JOINT_NAMES[0]} …）")


def check_policy_command_slot(report: Report) -> None:
    from bpx_rma import spec as rma_spec
    from bpx_traction import spec

    try:
        begin, end = spec.policy_command_slice()
    except KeyError as exc:
        report.fail("策略帧的 cmd 槽", str(exc))
        return

    if end - begin != 3:
        report.fail("策略帧的 cmd 槽", f"应为 3 维，实际 {end - begin}")
        return

    size = spec.policy_frame_size()
    detail = f"帧 {size} 维，cmd 在 [{begin}:{end}]"
    if size == 45:
        report.ok("策略帧的 cmd 槽", detail + "（基线）")
    elif size == 47:
        report.ok("策略帧的 cmd 槽", detail + "（相位进观测，新项追加在末尾，cmd 位置不变）")
    else:
        report.warn(
            "策略帧的 cmd 槽",
            detail + f"\nbpx_rma 的帧变成了 {size} 维 —— 牵引层本身不受影响"
                     "（只写 cmd 槽），但板子契约、历史维数、导出都要跟着复核",
        )


# ---------------------------------------------------------------------------
# 闸门三：映射参数
# ---------------------------------------------------------------------------
def check_mapping_params(report: Report) -> None:
    from bpx_traction import spec
    from bpx_traction.mapping import DEFAULT_PARAMS_PATH, AdmittanceParams

    params = (
        AdmittanceParams.from_json(DEFAULT_PARAMS_PATH)
        if DEFAULT_PARAMS_PATH.exists()
        else AdmittanceParams()
    )
    source = DEFAULT_PARAMS_PATH.name if DEFAULT_PARAMS_PATH.exists() else "代码默认值"

    try:
        params.validate()
    except ValueError as exc:
        report.fail("导纳参数", f"{source}：{exc}")
        return

    saturation = params.saturation_force_n()
    design_max = spec.FORCE_DESIGN_RANGE_N[1]
    detail = (
        f"来源 {source}\n"
        f"M = {params.virtual_mass_kg:.2f} kg   B = {params.virtual_damping_ns_per_m:.2f} N·s/m   "
        f"τ = {params.tau_s() * 1000:.0f} ms\n"
        f"{saturation:.1f} N 时指令到 vx 上限（设计范围上限 {design_max:.0f} N）"
    )
    if abs(saturation - design_max) > 0.5:
        report.warn(
            "导纳参数",
            detail + f"\nB 和设计意图对不上，用 "
                     f"AdmittanceParams.damping_for_saturation({design_max:g}, "
                     f"{params.vx_range[1]:g}, {params.deadzone_off_n:g}, "
                     f"{params.min_speed_mps:g}) 重算",
        )
    else:
        report.ok("导纳参数", detail)

    if not params.calibrated:
        report.warn(
            "死区仍是占位值（真机量过，现有估计器定不出来）",
            f"进 {params.deadzone_on_n:.1f} N / 出 {params.deadzone_off_n:.1f} N 是占位值。\n"
            "2026-10-26 真机 300 s 无外力行走实测：误报力 p99 39–46 N、max 50–54 N，\n"
            f"而设计力范围上限只有 {spec.FORCE_DESIGN_RANGE_N[1]:.0f} N —— 没有阈值能同时\n"
            "放过真牵引、挡住噪声。先把估计器的 sim2real 做掉。\n"
            "`calibrated=False` 会让 TractionRuntime.preflight() 拒绝启用牵引。",
        )

    if params.min_speed_mps <= params.stand_threshold_mps:
        report.fail(
            "起步速度跨过 stand_threshold",
            f"min_speed({params.min_speed_mps}) ≤ stand_threshold({params.stand_threshold_mps})，"
            "相位观测会在阈值附近抖",
        )
    else:
        report.ok(
            "起步速度跨过 stand_threshold",
            f"{params.min_speed_mps:.2f} > {params.stand_threshold_mps:.2f}",
        )


def check_gait_stand_threshold(report: Report) -> None:
    """映射里记的 stand_threshold 必须和 `bpx_rma.mdp.gait.gait_phase_obs` 的默认值一致。

    `gait.py` 不 import isaaclab 的重模块时可以直接读默认参数；读不到就退化成
    源码里抓一下，再不行就跳过（全量档会用真实 cfg 复核）。
    """
    import inspect

    from bpx_traction.mapping import DEFAULT_PARAMS_PATH, AdmittanceParams

    params = (
        AdmittanceParams.from_json(DEFAULT_PARAMS_PATH)
        if DEFAULT_PARAMS_PATH.exists()
        else AdmittanceParams()
    )
    try:
        from bpx_rma.mdp import gait

        signature = inspect.signature(gait.gait_phase_obs)
        actual = signature.parameters["stand_threshold"].default
    except Exception as exc:        # gait.py 需要 isaaclab 时会落到这里
        report.skip("gait stand_threshold 一致", f"读不到（{type(exc).__name__}），全量档会查")
        return

    if abs(actual - params.stand_threshold_mps) > 1e-9:
        report.fail(
            "gait stand_threshold 一致",
            f"gait_phase_obs 用的是 {actual}，映射里记的是 {params.stand_threshold_mps}。\n"
            "两者必须一致，否则 min_speed 跨不过真正的阈值。",
        )
    else:
        report.ok("gait stand_threshold 一致", f"两边都是 {actual}")


# ---------------------------------------------------------------------------
# 闸门四：模型与导出
# ---------------------------------------------------------------------------
def check_model(report: Report) -> None:
    try:
        import torch
    except ImportError:
        report.skip("模型结构", "没装 torch")
        return

    from bpx_traction import spec
    from bpx_traction.model import ForceTCN, NormalizedForceTCN, count_parameters

    net = ForceTCN().eval()
    x = torch.randn(2, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    values, confidence = net(x)

    if values.shape[1] != spec.OUTPUT_SIZE or confidence.shape[1] != spec.CONFIDENCE_SIZE:
        report.fail("模型结构", f"输出维数不对：{values.shape} / {confidence.shape}")
        return
    cycles = net.gait_cycles_visible()
    report.ok(
        "模型结构",
        f"{count_parameters(net):,} 参数，空洞率 {net.dilations}\n"
        f"感受野 {net.receptive_field()} 帧 = {net.receptive_field_s() * 1000:.0f} ms "
        f"= {cycles:.2f} 个步态周期",
    )
    if cycles < 1.0:
        report.warn(
            "感受野覆盖步态周期",
            f"只覆盖 {cycles:.2f} 个周期 —— 网络看不到周期性模式，分不清"
            f"'支撑相在蹬地'和'有外力'。实测全 1 空洞比 (1,2,4,8) 的 "
            f"xy RMSE 差 10.8%、方向差 15.9%。",
        )
    else:
        report.ok("感受野覆盖步态周期", f"{cycles:.2f} 个（步态周期 0.45 s）")

    # 因果性：改动感受野之外的老帧，输出必须逐位不变。
    # 默认空洞下感受野 31 帧 > 窗口 25 帧，没有"外面"的帧可改，所以这条结构性质
    # 用非空洞版来验（感受野 9 < 25）。两者共用同一个 CausalBlock。
    # 这里验的是 **padding 只补左边**，和归一化方式无关，所以固定用 norm="layer" 跑。
    # norm="group" 的统计量跨时间池化，整窗都会影响输出 —— 那不是 padding 的问题，
    # 部署时也仍然因果（窗口末端就是"现在"），见 model.py 的模块文档。
    plain = ForceTCN(dilations=None, norm="layer").eval()
    outside = spec.HISTORY_LENGTH - plain.receptive_field()
    with torch.no_grad():
        base_v, base_c = plain(x)
        x2 = x.clone()
        x2[:, :outside] = torch.randn(2, outside, spec.FRAME_SIZE) * 50.0
        v2, c2 = plain(x2)
    if torch.equal(base_v, v2) and torch.equal(base_c, c2):
        report.ok("因果卷积", f"非空洞版：最老的 {outside} 帧不影响输出（padding 只补左边）")
    else:
        report.fail("因果卷积", "感受野之外的帧影响了输出 —— padding 不是因果的")

    # 输出量程
    extreme, _ = net(torch.full((1, spec.HISTORY_LENGTH, spec.FRAME_SIZE), 1e4))
    bad = []
    for (name, _), (begin, end) in zip(spec.OUTPUT_TERMS, spec.output_slices()):
        act, scale, offset = spec.OUTPUT_ACTIVATIONS[name]
        lo, hi = (-scale, scale) if act == "tanh" else (offset, offset + scale)
        chunk = extreme[:, begin:end]
        if chunk.min() < lo - 1e-4 or chunk.max() > hi + 1e-4:
            bad.append(f"{name}: [{chunk.min():.3f}, {chunk.max():.3f}] 越出 [{lo}, {hi}]")
    if bad:
        report.fail("输出量程", "\n".join(bad))
    else:
        report.ok("输出量程", f"力头钳在 ±{spec.FORCE_SCALE_N:g} N，高度头 0.20~0.50 m")

    # TorchScript 一致性
    wrapped = NormalizedForceTCN(net, torch.zeros(spec.FRAME_SIZE), torch.ones(spec.FRAME_SIZE)).eval()
    traced = torch.jit.trace(wrapped, torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE))
    with torch.no_grad():
        ev, _ = wrapped(x)
        tv, _ = traced(x)
    err = float((ev - tv).abs().max())
    if err < 1e-5:
        report.ok("TorchScript 一致", f"最大偏差 {err:.2e}")
    else:
        report.fail("TorchScript 一致", f"最大偏差 {err:.2e} 超过 1e-5")


# ---------------------------------------------------------------------------
# 闸门五（全量档）：映射限幅 == 当前任务的命令区间
# ---------------------------------------------------------------------------
def check_command_ranges(report: Report, task: str) -> None:
    """**这是 git 最抓不到、后果最静默的一条。**

    映射把输出夹在 `vx_range` / `vy_range` 内，而这两个数必须等于策略**训练时见过的**
    命令分布。底层那边调一下 `commands.base_velocity.ranges`，文件层面和牵引毫无冲突，
    但牵引就会开始发训练分布之外的指令 —— 策略在那里的行为无定义。
    """
    from isaaclab_tasks.utils import load_cfg_from_registry

    import bpx_rma  # noqa: F401  注册任务
    from bpx_traction.mapping import DEFAULT_PARAMS_PATH, AdmittanceParams

    params = (
        AdmittanceParams.from_json(DEFAULT_PARAMS_PATH)
        if DEFAULT_PARAMS_PATH.exists()
        else AdmittanceParams()
    )

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    ranges = cfg.commands.base_velocity.ranges
    actual_x = tuple(float(v) for v in ranges.lin_vel_x)
    actual_y = tuple(float(v) for v in ranges.lin_vel_y)

    problems = []
    if tuple(params.vx_range) != actual_x:
        problems.append(f"vx: 映射 {tuple(params.vx_range)}  任务 {actual_x}")
    if tuple(params.vy_range) != actual_y:
        problems.append(f"vy: 映射 {tuple(params.vy_range)}  任务 {actual_y}")
    if params.source_task != task:
        problems.append(f"source_task: 映射记的是 {params.source_task}，这次查的是 {task}")

    if problems:
        report.fail(
            "映射限幅 == 训练分布",
            "\n".join(problems)
            + f"\n改 {DEFAULT_PARAMS_PATH.name} 的 vx_range / vy_range / source_task，"
              "然后用 damping_for_saturation 重算 B",
        )
    else:
        report.ok("映射限幅 == 训练分布", f"{task}   vx {actual_x}   vy {actual_y}")

    # 顺带把真实 cfg 里的 stand_threshold 也核一遍
    phase = getattr(cfg.observations.policy, "phase", None)
    if phase is None:
        report.ok("策略帧无相位项", f"{task} 用的是基线帧（{cfg.observations.policy.__class__.__name__}）")
    else:
        actual = phase.params.get("stand_threshold", 0.1)
        if abs(actual - params.stand_threshold_mps) > 1e-9:
            report.fail(
                "gait stand_threshold 一致",
                f"任务里是 {actual}，映射里记的是 {params.stand_threshold_mps}",
            )
        else:
            report.ok("gait stand_threshold 一致", f"任务和映射都是 {actual}")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="牵引层契约闸门")
    parser.add_argument(
        "--with-env", action="store_true",
        help="额外启 Isaac Sim，核对映射限幅 == 任务的命令区间（改过命令区间后必跑）",
    )
    parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-BpxPro-v0")
    known, rest = parser.parse_known_args()

    simulation_app = None
    if known.with_env:
        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(launcher_parser)
        launcher_parser.set_defaults(headless=True)
        app_args = launcher_parser.parse_args(rest)
        simulation_app = AppLauncher(app_args).app

    report = Report()
    check_bpx_rma_untouched(report)
    check_spec_self_consistent(report)
    check_joint_order(report)
    check_policy_command_slot(report)
    check_mapping_params(report)
    check_gait_stand_threshold(report)
    check_model(report)

    if known.with_env:
        check_command_ranges(report, known.task)
    else:
        report.skip("映射限幅 == 训练分布", "要 --with-env（启 Isaac Sim）才能查")

    print("\n" + report.render())

    if simulation_app is not None:
        simulation_app.close()

    if report.failed:
        print("\n契约不符，先修。")
        return 1
    print("\n契约通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
