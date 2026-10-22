#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class StateFusion(nn.Module):
    def __init__(self, primary: torch.jit.ScriptModule, state: torch.jit.ScriptModule):
        super().__init__()
        self.primary = primary
        self.state = state

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        primary_values, primary_confidence = self.primary(history)
        state_values, _ = self.state(history)
        values = torch.cat((state_values[:, :4], primary_values[:, 4:]), dim=1)
        return values, primary_confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    primary = torch.jit.load(str(Path(args.primary) / "estimator.ts"), map_location="cpu").eval()
    state = torch.jit.load(str(Path(args.state) / "estimator.ts"), map_location="cpu").eval()
    model = StateFusion(primary, state).eval()
    example = torch.zeros(1, 25, 64)
    torch.jit.trace(model, example).save(str(output / "estimator.ts"))
    (output / "fusion_report.json").write_text(
        json.dumps({"schema_version": "gate41-state-fusion-v1", "primary": args.primary, "state_specialist": args.state, "replaced_outputs": ["base_velocity_w", "base_height"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
