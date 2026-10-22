#!/usr/bin/env python3
"""Acceptance analysis for v19 dense directions and direct force jumps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def analyze_run(run: Path) -> dict:
    manifest = json.loads(next(run.glob("manifest_seed*.json")).read_text())
    data = np.load(run / manifest["file"])
    rollout = data["estimator_in_loop_rollout"]
    diag = data["extended_diagnostics"]
    scenario_idx = rollout[:, -2].astype(np.int64)
    rows = []
    for idx, scenario in enumerate(manifest["scenarios"]):
        mask = scenario_idx == idx
        rr, dd = rollout[mask], diag[mask]
        time_s = rr[:, 2]
        q = dd[:, :4]
        w, x, y, z = q.T
        roll = np.degrees(np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))
        stable = bool(not np.any(rr[:, 3] > 0.5) and rr[:, 16].min() >= 0.16 and
                      max(np.abs(roll).max(), np.abs(pitch).max()) <= 60.0)
        if scenario["family"] == "jump":
            window = time_s >= 4.75
        elif scenario.get("release"):
            window = (time_s >= 1.5) & (time_s < 3.8)
        else:
            window = time_s >= 2.0
        force_xy = rr[window, 4:6]
        velocity_xy = rr[window, 17:19]
        hmag = np.linalg.norm(force_xy, axis=1)
        speed = np.linalg.norm(velocity_xy, axis=1)
        projection = np.sum(velocity_xy * force_xy, axis=1) / np.maximum(hmag, 1e-6)
        moving = (hmag >= 8.0) & (speed >= 0.02)
        cosine = np.sum(velocity_xy * force_xy, axis=1) / np.maximum(speed * hmag, 1e-6)
        mean_projection = float(np.mean(projection[hmag >= 8.0])) if np.any(hmag >= 8.0) else None
        mean_cosine = float(np.mean(cosine[moving])) if np.any(moving) else None
        # The deployed follow command deliberately has a deadband at small
        # loads, so a stable 5-10 N scene may remain nearly stationary. Above
        # that deadband, require a positive net velocity projection; 5 mm/s
        # is used as a conservative numerical-motion threshold rather than
        # demanding a large displacement from a short 8-second rollout.
        follows = True if not np.any(hmag >= 8.0) else bool(mean_projection is not None and mean_projection >= 0.005)
        release_ok = True
        if scenario.get("release"):
            release_ok = bool(np.linalg.norm(rr[time_s >= 7.0, 7:10], axis=1).mean() <= 10.0)
        passed = stable and follows and release_ok
        rows.append({
            "scenario_index": idx, "scenario_id": scenario["id"], "family": scenario["family"],
            "stable": stable, "follows_new_force": follows, "release_estimate_cleared": release_ok,
            "mean_velocity_projection_mps": mean_projection, "mean_direction_cosine": mean_cosine,
            "min_height_m": float(rr[:, 16].min()),
            "max_tilt_abs_deg": float(max(np.abs(roll).max(), np.abs(pitch).max())),
            "passed": passed,
        })
    return {"seed": manifest["seed"], "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    runs = []
    for category in ("nominal", "robust"):
        for path in sorted((root / category).glob("seed*")):
            if list(path.glob("manifest_seed*.json")):
                result = analyze_run(path)
                result["category"] = category
                runs.append(result)
    summary = {}
    for category in ("nominal", "robust"):
        rows = [row for run in runs if run["category"] == category for row in run["rows"]]
        families = {}
        for family in ("horizontal", "vertical", "combined", "jump"):
            sub = [row for row in rows if row["family"] == family]
            families[family] = {"total": len(sub), "passed": sum(r["passed"] for r in sub),
                                "pass_rate": sum(r["passed"] for r in sub) / len(sub) if sub else None}
        summary[category] = {"total": len(rows), "passed": sum(r["passed"] for r in rows),
                             "pass_rate": sum(r["passed"] for r in rows) / len(rows) if rows else 0.0,
                             "by_family": families}
    nominal_ok = summary["nominal"]["pass_rate"] >= 0.95
    robust_ok = summary["robust"]["pass_rate"] >= 0.90
    output = {"schema_version": "continuous-direction-jump-certification-v19",
              "acceptance": {"nominal_min": 0.95, "robust_min": 0.90,
                             "min_velocity_projection_mps": 0.005,
                             "scope": "horizontal 0..355 degrees at 5-degree spacing; direct jumps at 10-degree spacing; MJLab only"},
              "status": "PASSED" if nominal_ok and robust_ok else "FAILED",
              "summary": summary, "runs": runs}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
