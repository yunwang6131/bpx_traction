#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from world_force.model import MissingStateTCN


class Normalized(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model((history - self.mean) / self.std)


class StateFusion(nn.Module):
    def __init__(self, primary: nn.Module, state: nn.Module):
        super().__init__()
        self.primary = primary
        self.state = state

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        primary_values, primary_confidence = self.primary(history)
        state_values, _ = self.state(history)
        return torch.cat((state_values[:, :4], primary_values[:, 4:]), dim=1), primary_confidence


def load_primary(path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path / "estimator.pt", map_location="cpu")
    model = MissingStateTCN(input_dim=checkpoint["input_dim"], channels=checkpoint.get("channels", 64)).eval()
    model.load_state_dict(checkpoint["model"])
    return Normalized(model, checkpoint["mean"], checkpoint["std"]).eval(), checkpoint


def load_state(path: Path) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path / "state.pt", map_location="cpu")
    model = MissingStateTCN(input_dim=checkpoint["input_dim"], channels=checkpoint["channels"]).eval()
    model.load_state_dict(checkpoint["model"])
    return Normalized(model, checkpoint["mean"], checkpoint["std"]).eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    primary, primary_checkpoint = load_primary(Path(args.primary))
    state, state_checkpoint = load_state(Path(args.state))
    model = StateFusion(primary, state).eval()
    example = torch.zeros(1, 25, primary_checkpoint["input_dim"])
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
    np.savez(
        output / "normalization.npz",
        primary_mean=primary_checkpoint["mean"].numpy(),
        primary_std=primary_checkpoint["std"].numpy(),
        state_mean=state_checkpoint["mean"].numpy(),
        state_std=state_checkpoint["std"].numpy(),
    )
    (output / "fusion_report.json").write_text(
        json.dumps({"schema_version": "gate41-state-fusion-v2", "primary": args.primary, "state_specialist": args.state, "replaced_outputs": ["base_velocity_w", "base_height"]}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
