#!/usr/bin/env python3
"""把多个种子的估计器打包成一个平均集成，导出 TorchScript / ONNX。

    python scripts/export_ensemble.py \
        --models logs/traction/sweep1/a1_res128 logs/traction/sweep1/a1_res128_s1 \
                 logs/traction/sweep1/a1_res128_s2 \
        --out logs/traction/estimator_ensemble

为什么值得：三种子平均在测试集上 xy RMSE −10%、零力误报 p95 −19%（sweep1 实测），
代价只是参数 ×N 和推理 ×N —— 对 55k–210k 的小模型，板上 CPU 预算（p99 < 6 ms）
吃得下，本脚本当场实测并写进报告。

输出和单模型完全同构（values (B,15) + confidence (B,4)，归一化焊在图里），
`play_traction.py --estimator` 和板端 runtime 直接换路径就能用。

约束：各成员的 normalization.npz 必须逐位一致（同一份数据集训出来的才行）——
不一致说明它们见过的输入分布不同，平均没有意义，直接拒绝。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_traction import spec  # noqa: E402
from bpx_traction.model import NormalizedForceTCN, count_parameters, from_checkpoint  # noqa: E402
from bpx_traction.normalize import Normalizer  # noqa: E402


class EnsembleTCN(nn.Module):
    """成员输出的算术平均。values 的各头都过了有界激活，平均后仍在量程内。"""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values, confidence = zip(*[m(history) for m in self.members])
        return (torch.stack(values).mean(dim=0),
                torch.stack(confidence).mean(dim=0))


def main() -> None:
    ap = argparse.ArgumentParser(description="多种子估计器集成导出")
    ap.add_argument("--models", type=Path, nargs="+", required=True,
                    help="成员目录（各含 estimator.pt + normalization.npz）")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    members, scores = [], []
    normalizer = Normalizer.load(args.models[0] / "normalization.npz")
    for d in args.models:
        nz = Normalizer.load(d / "normalization.npz")
        if not (np.array_equal(nz.mean, normalizer.mean)
                and np.array_equal(nz.std, normalizer.std)):
            raise SystemExit(f"{d} 的归一化和 {args.models[0]} 不一致 —— "
                             f"不是同一份数据训的，平均没有意义")
        ck = torch.load(d / "estimator.pt", map_location="cpu", weights_only=False)
        members.append(from_checkpoint(ck))
        scores.append(float(ck.get("score_xy_rmse_n", float("nan"))))

    wrapped = NormalizedForceTCN(
        EnsembleTCN(members),
        torch.from_numpy(normalizer.mean), torch.from_numpy(normalizer.std),
    ).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    traced = torch.jit.trace(wrapped, example)
    traced.save(str(args.out / "estimator.ts"))
    torch.onnx.export(
        wrapped, example, args.out / "estimator.onnx",
        input_names=["sensor_history"], output_names=["values", "confidence"],
        dynamic_axes={"sensor_history": {0: "batch"}, "values": {0: "batch"},
                      "confidence": {0: "batch"}},
        opset_version=17,
    )
    normalizer.save(args.out / "normalization.npz")

    with torch.no_grad():
        probe = torch.randn(64, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
        ev, ec = wrapped(probe)
        tv, tc = traced(probe)
    parity = max(float((ev - tv).abs().max()), float((ec - tc).abs().max()))

    # 板上跑的是单帧推理，按 batch=1 在 CPU 上量延迟
    single = torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    with torch.no_grad():
        for _ in range(20):
            traced(single)
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            traced(single)
            times.append((time.perf_counter() - t0) * 1000)
    p99 = float(np.percentile(times, 99))

    report = {
        "members": [str(d) for d in args.models],
        "member_val_xy_rmse_n": scores,
        "parameters": count_parameters(wrapped),
        "torchscript_parity_max_abs": parity,
        "cpu_latency_ms": {"p50": float(np.percentile(times, 50)), "p99": p99,
                           "budget_ms": 6.0, "pass": bool(p99 < 6.0)},
    }
    (args.out / "ensemble_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[traction] {len(members)} 成员集成 → {args.out}")
    print(f"           参数 {report['parameters']:,}   parity {parity:.2e}   "
          f"CPU p99 {p99:.2f} ms（预算 6 ms → {'过' if p99 < 6 else '未过'}）")


if __name__ == "__main__":
    main()
