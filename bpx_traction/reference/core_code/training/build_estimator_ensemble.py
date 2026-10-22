#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class Ensemble(nn.Module):
    def __init__(self, models: list[torch.jit.ScriptModule]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = []
        confidence = []
        for model in self.models:
            value, conf = model(history)
            values.append(value)
            confidence.append(conf)
        return torch.stack(values).mean(0), torch.stack(confidence).mean(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    members = [torch.jit.load(str(Path(path) / "estimator.ts"), map_location="cpu").eval() for path in args.models]
    model = Ensemble(members).eval()
    example = torch.zeros(1, 25, 64)
    torch.jit.trace(model, example).save(str(output / "estimator.ts"))
    torch.onnx.export(
        model,
        example,
        output / "estimator.onnx",
        input_names=["sensor_history"],
        output_names=["values", "confidence"],
        dynamic_axes={"sensor_history": {0: "batch"}, "values": {0: "batch"}, "confidence": {0: "batch"}},
        opset_version=17,
    )
    (output / "ensemble_report.json").write_text(
        json.dumps({"schema_version": "gate41-estimator-ensemble-v1", "members": args.models, "aggregation": "arithmetic_mean"}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
