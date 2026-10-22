#!/usr/bin/env python3
"""Behavioral certification for the 192-scene v20 direction matrix.

Unlike the legacy completion check, this verifier does not count a rollout as
successful merely because the robot did not terminate.  It checks stability,
motion along the requested world-frame horizontal force, and stopping after a
release for every scenario and seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def attitude_deg(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = q.T
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def inspect(run: Path, required: int) -> dict:
    manifest_path = next(run.glob("manifest_seed*.json"))
    manifest = json.loads(manifest_path.read_text())
    arrays = np.load(run / manifest["file"])
    rollout = arrays["estimator_in_loop_rollout"]
    diagnostics = arrays["extended_diagnostics"]
    rows = []

    for idx, scenario in enumerate(manifest["scenarios"]):
        mask = rollout[:, -2].astype(np.int64) == idx
        r = rollout[mask]
        d = diagnostics[mask]
        t = r[:, 2]
        force_xy = r[:, 4:6]
        vel_xy = r[:, 17:19]
        requested_xy = np.asarray(scenario["force_w_n"][:2], dtype=np.float64)
        requested_mag = float(np.linalg.norm(requested_xy))
        speed = np.linalg.norm(vel_xy, axis=1)
        release_at = float(scenario.get("release_at_s", 1.0e9))

        # Ignore the force-estimator/gait transient at force onset.  For
        # release cases, measure tracking only before the release transition.
        active = (t >= 0.75) & (t < release_at - 0.25)
        direction = requested_xy / max(requested_mag, 1.0e-6)
        projection = vel_xy @ direction if requested_mag > 0 else np.zeros(len(r))
        active_projection = float(projection[active].mean()) if active.any() else None
        active_positive_fraction = (
            float((projection[active] > 0.0).mean()) if active.any() else None
        )

        final = t >= t[-1] - 1.0
        final_speed = float(speed[final].mean())
        roll, pitch = attitude_deg(d[:, :4])
        horizontal_follow_required = requested_mag >= 8.0
        horizontal_stand_required = requested_mag < 8.0
        released = bool(scenario.get("release", False))

        checks = {
            "samples_present": len(r) > 0,
            "no_termination": not bool((r[:, 3] > 0.5).any()),
            "height_ok": float(r[:, 16].min()) >= 0.16,
            "attitude_ok": max(float(np.abs(roll).max()), float(np.abs(pitch).max())) <= 45.0,
            "active_direction_ok": (
                True
                if not horizontal_follow_required
                else active_projection is not None
                and active_projection >= 0.03
                and active_positive_fraction is not None
                and active_positive_fraction >= 0.60
            ),
            # Pure vertical and sub-dead-zone pulls must not manufacture a
            # horizontal walking command.
            "subthreshold_or_vertical_stand_ok": (
                True if not horizontal_stand_required else float(speed.mean()) <= 0.15
            ),
            "release_stop_ok": True if not released else final_speed <= 0.15,
        }
        rows.append(
            {
                "scenario_index": idx,
                "scenario_id": scenario["id"],
                "family": scenario.get("family"),
                "passed": all(checks.values()),
                "checks": checks,
                "active_projection_mps": active_projection,
                "active_positive_fraction": active_positive_fraction,
                "mean_speed_mps": float(speed.mean()),
                "final_horizontal_speed_mps": final_speed,
                "min_height_m": float(r[:, 16].min()),
                "max_roll_abs_deg": float(np.abs(roll).max()),
                "max_pitch_abs_deg": float(np.abs(pitch).max()),
            }
        )

    passed = sum(row["passed"] for row in rows)
    return {
        "seed": int(manifest["seed"]),
        "actor_checkpoint": manifest["actor_checkpoint"],
        "passed": passed,
        "total": len(rows),
        "status": "PASSED" if passed == required and len(rows) == required else "FAILED",
        "failed_scenarios": [row["scenario_id"] for row in rows if not row["passed"]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-per-seed", type=int, default=192)
    args = parser.parse_args()
    root = Path(args.root)
    if args.required_per_seed <= 0:
        raise ValueError("--required-per-seed must be positive")
    runs = [inspect(path, args.required_per_seed) for path in sorted(root.glob("seed*")) if list(path.glob("manifest_seed*.json"))]
    result = {
        "schema_version": "v20-direction-matrix-behavior-certification-v1",
        "acceptance": {
            "required_per_seed": f"{args.required_per_seed}/{args.required_per_seed}",
            "active_projection_mps_min": 0.03,
            "active_positive_fraction_min": 0.60,
            "released_or_nonhorizontal_speed_mps_max": 0.15,
            "min_height_m": 0.16,
            "max_abs_roll_pitch_deg": 45.0,
        },
        "runs": runs,
        "status": "PASSED" if runs and all(run["status"] == "PASSED" for run in runs) else "FAILED",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(result["status"], [(run["seed"], run["passed"], run["total"]) for run in runs])


if __name__ == "__main__":
    main()
