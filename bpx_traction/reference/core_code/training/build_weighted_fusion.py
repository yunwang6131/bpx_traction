#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class WeightedFusion(nn.Module):
    def __init__(self, primary_a, primary_b, state_a, state_b, primary_weight: float, state_weight: float):
        super().__init__()
        self.primary_a = primary_a
        self.primary_b = primary_b
        self.state_a = state_a
        self.state_b = state_b
        self.primary_weight = primary_weight
        self.state_weight = state_weight

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pa, ca = self.primary_a(history)
        pb, cb = self.primary_b(history)
        sa, _ = self.state_a(history)
        sb, _ = self.state_b(history)
        primary = (1.0 - self.primary_weight) * pa + self.primary_weight * pb
        confidence = (1.0 - self.primary_weight) * ca + self.primary_weight * cb
        state = (1.0 - self.state_weight) * sa + self.state_weight * sb
        return torch.cat((state[:, :4], primary[:, 4:]), 1), confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-a", required=True)
    parser.add_argument("--primary-b", required=True)
    parser.add_argument("--state-a", required=True)
    parser.add_argument("--state-b", required=True)
    parser.add_argument("--primary-weight", type=float, required=True)
    parser.add_argument("--state-weight", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    load = lambda path: torch.jit.load(str(Path(path) / "estimator.ts"), map_location="cpu").eval()
    model = WeightedFusion(load(args.primary_a), load(args.primary_b), load(args.state_a), load(args.state_b), args.primary_weight, args.state_weight).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.jit.trace(model, torch.zeros(1, 25, 64)).save(str(output / "estimator.ts"))
    (output / "fusion_report.json").write_text(json.dumps(vars(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
