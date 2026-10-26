#!/usr/bin/env python3
"""Fail-closed contract check for BPX velocity-RMA deployment."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT / "deploy/robots/bpx/config/policy/velocity_rma/v0/exported/policy.onnx"
DEPLOY = ROOT / "deploy/robots/bpx/config/policy/velocity_rma/v0/params/deploy.yaml"
JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("fl", "fr", "hl", "hr")
    for joint in ("hip_roll", "hip_pitch", "knee")
]
Q0 = [0.0, 0.7, -1.4] * 4
OBS_NAMES = [
    "joint_pos", "joint_vel", "base_ang_vel",
    "projected_gravity", "last_action", "command",
]
OBS_DIMS = [12, 12, 3, 3, 12, 3]


def fail(message: str) -> None:
    raise ValueError(message)


def equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        fail(f"{label}: expected {expected!r}, got {actual!r}")


def close(label: str, actual: Any, expected: Any, tol: float = 1e-6) -> None:
    try:
        av = [float(actual)] if not isinstance(actual, list) else [float(x) for x in actual]
        ev = [float(expected)] if not isinstance(expected, list) else [float(x) for x in expected]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: non-numeric value") from exc
    if len(av) != len(ev) or not all(math.isfinite(x) for x in av + ev):
        fail(f"{label}: invalid vector")
    if any(abs(a - e) > tol for a, e in zip(av, ev)):
        fail(f"{label}: expected {ev!r}, got {av!r}")


def cap_ok(label: str, actual: Any, limit: float) -> None:
    """手柄 caps 范围检查：接受任何 [neg, pos] 对称区间，只要 |两端| <= limit。

    放开让 v4 所有策略(deploy.yaml 里 caps 写 3/5/6 都行)都能上板；手柄实际
    上限由 bpx_dual_pad.py 的 POLICY_VX_FWD/BACK_MAX(=6.0) 决定，不是这里声明。
    """
    try:
        av = [float(x) for x in actual]
    except (TypeError, ValueError) as exc:
        fail(f"{label}: non-numeric {actual!r}")
        raise exc
    if len(av) != 2 or av[0] > 1e-6 or av[1] < -1e-6 or max(abs(av[0]), abs(av[1])) > limit + 1e-6:
        fail(f"{label}: {av!r} 不是 [neg,pos] 或超出 +/-{limit}")


def shape(value_info: Any, label: str, onnx_module: Any) -> list[int]:
    tensor = value_info.type.tensor_type
    equal(f"{label} dtype", tensor.elem_type, onnx_module.TensorProto.FLOAT)
    result = []
    for index, dim in enumerate(tensor.shape.dim):
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            fail(f"{label} dimension {index} is not fixed positive")
        result.append(int(dim.dim_value))
    return result


def metadata_map(model: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in model.metadata_props:
        if item.key in result:
            fail(f"duplicate ONNX metadata key {item.key!r}")
        result[item.key] = item.value
    return result


def csv_float(value: str) -> list[float]:
    return [float(x) for x in value.split(",")]


def validate_onnx() -> None:
    try:
        import onnx
    except ImportError:
        fail("Python package 'onnx' is required for the full policy contract check")
    if not ONNX.is_file():
        fail(f"RMA policy missing: {ONNX}")
    model = onnx.load(ONNX, load_external_data=False)
    inputs = {item.name: item for item in model.graph.input}
    equal("ONNX input names", set(inputs), {"policy", "adaptation_history"})
    equal("policy input shape", shape(inputs["policy"], "policy", onnx), [1, 45])
    equal(
        "adaptation_history input shape",
        shape(inputs["adaptation_history"], "adaptation_history", onnx),
        [1, 1125],
    )
    equal("ONNX output count", len(model.graph.output), 1)
    out = model.graph.output[0]
    equal("ONNX output name", out.name, "actions")
    equal("ONNX output shape", shape(out, "actions", onnx), [1, 12])

    md = metadata_map(model)
    required = {
        "joint_names", "joint_stiffness", "joint_damping", "joint_effort_limit",
        "default_joint_pos", "observation_names", "action_scale",
        "adaptation_history_length", "adaptation_history_names",
        "action_clip",
    }
    missing = required - set(md)
    if missing:
        fail(f"missing ONNX metadata: {sorted(missing)}")
    equal("metadata joint_names", md["joint_names"].split(","), JOINT_NAMES)
    equal("metadata observation_names", md["observation_names"].split(","), OBS_NAMES)
    equal(
        "metadata adaptation_history_names",
        md["adaptation_history_names"].split(","),
        OBS_NAMES,
    )
    equal("metadata adaptation_history_length", md["adaptation_history_length"], "25")
    close("metadata stiffness", csv_float(md["joint_stiffness"]), [40.0] * 12)
    close("metadata damping", csv_float(md["joint_damping"]), [1.0] * 12)
    close("metadata effort limit", csv_float(md["joint_effort_limit"]), [30.0] * 12)
    close("metadata default_joint_pos", csv_float(md["default_joint_pos"]), Q0)
    action_scale = csv_float(md["action_scale"])
    if len(action_scale) == 1:
        action_scale *= 12
    # mjlab's metadata writer uses three decimals, so 0.1875 is serialized as
    # 0.188 even though the graph/config runtime scale remains exactly 0.1875.
    close("metadata action_scale", action_scale, [0.1875] * 12, tol=5.1e-4)
    close("metadata action_clip", float(md["action_clip"]), 4.0)


def validate_deploy() -> None:
    cfg = yaml.safe_load(DEPLOY.read_text())
    equal("policy_name", cfg.get("policy_name"), "BPX-Velocity-RMA")
    equal("hardware convention gate", cfg.get("hardware_joint_convention_confirmed"), True)
    equal("joint map", cfg.get("joint_ids_map"), list(range(12)))
    equal("joint names", cfg.get("joint_names"), JOINT_NAMES)
    close("step_dt", cfg.get("step_dt"), 0.02)
    equal("policy frequency", cfg.get("policy_frequency_hz"), 50)
    equal("joint send frequency", cfg.get("joint_send_frequency_hz"), 500)
    equal(
        "deploy ONNX inputs",
        cfg.get("onnx_inputs"),
        {
            "policy": {"shape": [1, 45], "dtype": "float32"},
            "adaptation_history": {"shape": [1, 1125], "dtype": "float32"},
        },
    )
    equal(
        "deploy ONNX output",
        cfg.get("onnx_output"),
        {"name": "actions", "shape": [1, 12], "dtype": "float32"},
    )
    history = cfg.get("history", {})
    equal("frame size", history.get("frame_size"), 45)
    equal("history length", history.get("length"), 25)
    close("history duration", history.get("duration_s"), 0.5)
    equal("history ONNX order", history.get("onnx_order"), "term_major_oldest_to_newest")
    equal("history reset", history.get("reset"), "repeat_current_frame")
    observations = cfg.get("observations", [])
    equal("observation names", [x.get("name") for x in observations], OBS_NAMES)
    equal("observation dims", [x.get("dim") for x in observations], OBS_DIMS)
    equal("single-frame dimension", sum(x.get("dim", 0) for x in observations), 45)
    close("observation scales", [x.get("scale") for x in observations], [1.0] * 6)
    ranges = cfg.get("commands", {}).get("base_velocity", {}).get("ranges", {})
    # caps 改范围检查：所有策略(3/5/6)都兼容，手柄上限由 bpx_dual_pad 的 POLICY_VX_MAX(6.0) 决定
    cap_ok("vx caps", ranges.get("lin_vel_x"), 6.0)
    cap_ok("vy caps", ranges.get("lin_vel_y"), 2.0)
    cap_ok("wz caps", ranges.get("ang_vel_z"), 3.0)
    control = cfg.get("control", {})
    close("control stiffness", control.get("stiffness"), [40.0] * 12)
    close("control damping", control.get("damping"), [1.0] * 12)
    close("control effort limit", control.get("effort_limit"), [30.0] * 12)
    close("control q0", control.get("default_joint_pos"), Q0)
    close("control action scale", control.get("action_scale"), 0.1875)
    close("control raw action clip", control.get("raw_action_clip"), [-4.0, 4.0])
    equal("control action type", control.get("action_type"), "joint_position_target")
    # 2026-10 patch (research policy swaps): the DR section describes the
    # TRAINING of the installed policy and legitimately differs between the
    # factory policy and the identified-params study policies. Mismatch with
    # the factory values is now a warning, not a failure. All contract-safety
    # checks (shapes, metadata, control constants, history) remain fail-closed.
    # Original behaviour preserved in verify_policy_contract.py.orig.
    dr = cfg.get("domain_randomization", {})
    if dr.get("knee_frictionloss_nm") != [5.0, 15.0] or dr.get("knee_frictionloss_independent") is not True:
        print("WARN: domain_randomization differs from the factory policy "
              "(expected for research policies; see params/deploy.yaml)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args()
    validate_deploy()
    if not args.config_only:
        validate_onnx()
    suffix = "config only" if args.config_only else "ONNX + config"
    print(f"PASS: BPX velocity-RMA contract verified ({suffix}; 45 + 1125 -> 12).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
