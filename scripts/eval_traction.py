#!/usr/bin/env python3
"""离线验收：把训好的估计器放到测试集上分层出报告。

    conda activate bpx_rma
    python scripts/eval_traction.py

不需要 Isaac Sim。默认读 `logs/traction/estimator/estimator.ts` 和
`logs/traction/dataset/test.npz`。

报什么：分量 / 幅值 / **xy 向量误差**（映射实际消费的那个量）/ xy 平面内的方向误差 /
零力误报 / 幅值的系统性缩放 / 置信度标定，并按力大小、方向、采集策略分层。

**不报"应用验收通过"** —— 应用预算（允许多大力误差、多长恢复时间、多大指令抖动）
还没定，所以这里只能给描述性结果加一条从导纳 B 折算出来的指令误差预算。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_traction import metrics, spec  # noqa: E402
from bpx_traction.mapping import AdmittanceMapping, AdmittanceParams  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="力估计器离线验收")
    parser.add_argument("--model", type=str, default=None,
                        help="TorchScript 路径，默认 logs/traction/estimator/estimator.ts")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = Path(args.model) if args.model else (
        spec.TRACTION_LOG_DIR / "estimator" / "estimator.ts")
    root = Path(args.dataset) if args.dataset else spec.TRACTION_LOG_DIR / "dataset"

    model = torch.jit.load(str(model_path), map_location=device).eval()
    data = np.load(root / f"{args.split}.npz")
    history, targets = data["history"], data["targets"]
    print(f"[traction] {model_path.name}  ×  {args.split} {len(history):,} 样本")

    values, confidence = [], []
    with torch.no_grad():
        for begin in range(0, len(history), args.batch):
            chunk = torch.from_numpy(history[begin:begin + args.batch]).to(device)
            v, c = model(chunk)
            values.append(v.cpu().numpy())
            confidence.append(c.cpu().numpy())
    values = np.concatenate(values)
    confidence = np.concatenate(confidence)

    fb, fe = spec.output_slice("force_b")
    pred_force, true_force = values[:, fb:fe], targets[:, fb:fe]

    extra = {}
    if "policy_id" in data.files:
        extra["policy_id"] = data["policy_id"]
    report = metrics.stratified(pred_force, true_force, extra)

    # 置信度标定：只看 force 那一维（唯一有消费端的）
    force_conf = confidence[:, spec.CONFIDENCE_TERMS.index("force")]
    error_n = np.linalg.norm(pred_force - true_force, axis=1)
    report["confidence_calibration"] = metrics.confidence_calibration(force_conf, error_n)

    params = AdmittanceParams.from_json() if (
        spec.PACKAGE_DIR / "data" / "admittance.json").exists() else AdmittanceParams()
    budget = metrics.command_budget(params.virtual_damping_ns_per_m)
    verdict = metrics.verdict(report, budget)

    print("\n" + metrics.render(report, verdict))

    cal = report["confidence_calibration"]
    print(f"\n置信度标定  秩相关 {cal['rank_correlation']:+.2f}  "
          f"{'✓ 单调（置信度越高误差越小）' if cal['monotonic'] else '⚠️ 不单调 —— 高置信度配大误差比估不准更危险'}")
    for row in cal["bins"][::3]:
        print(f"  conf≈{row['confidence_mean']:.2f}   误差均值 {row['error_mean_n']:.2f} N   "
              f"P95 {row['error_p95_n']:.2f} N   n={row['n']:,}")

    # 把力误差过一遍映射，看它实际会造成多大的指令误差 —— 这才是下游真正关心的
    mapping = AdmittanceMapping(params)
    steady_true = np.array([mapping.params.steady_speed(float(m))
                            for m in np.linalg.norm(true_force[:, :2], axis=1)])
    steady_pred = np.array([mapping.params.steady_speed(float(m))
                            for m in np.linalg.norm(pred_force[:, :2], axis=1)])
    command_error = np.abs(steady_pred - steady_true)
    print(f"\n折算成指令误差（稳态，未计滤波和限幅）")
    print(f"  均值 {command_error.mean():.4f}   P95 {np.percentile(command_error, 95):.4f}   "
          f"最大 {command_error.max():.4f} m/s")
    print(f"  参照：策略自身的跟踪误差 {metrics.POLICY_TRACKING_ERROR_MPS:.3f} m/s")
    report["command_error_mps"] = {
        "mean": float(command_error.mean()),
        "p95": float(np.percentile(command_error, 95)),
        "max": float(command_error.max()),
        "reference_tracking_error": metrics.POLICY_TRACKING_ERROR_MPS,
    }
    report["verdict"] = verdict
    report["model"] = str(model_path)
    report["split"] = args.split
    report["admittance"] = {"B": params.virtual_damping_ns_per_m,
                            "M": params.virtual_mass_kg,
                            "budget_n": budget}

    out = Path(args.output) if args.output else (
        spec.TRACTION_LOG_DIR / "estimator" / f"acceptance_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[traction] 报告 → {out}")


if __name__ == "__main__":
    main()
