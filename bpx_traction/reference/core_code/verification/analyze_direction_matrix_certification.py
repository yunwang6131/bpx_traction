#!/usr/bin/env python3
"""Summarize all-direction hold/release rollouts without oracle signals."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def one(run: Path) -> dict:
    manifest = json.loads(next(run.glob("manifest_seed*.json")).read_text())
    data = np.load(run / manifest["file"])
    r = data["estimator_in_loop_rollout"]
    d = data["extended_diagnostics"]
    # fixed rollout schema written by run_direction_matrix_rollouts.py
    # seed,step,time,terminated, true(3), estimate(3), confidence(4),
    # pose(3), velocity(3), action(12), force_b(3), scenario_index, sample_index
    terminated = r[:, 3] > 0.5
    scenario_idx = r[:, -2].astype(np.int64)
    scenario_by_idx = {i: s for i, s in enumerate(manifest["scenarios"])}
    rows = []
    for idx, scenario in scenario_by_idx.items():
        mask = scenario_idx == idx
        if not mask.any():
            continue
        rr = r[mask]
        dd = d[mask]
        force_err = np.linalg.norm(rr[:, 7:10] - rr[:, 4:7], axis=1)
        q = dd[:, :4]
        w, x, y, z = q.T
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        rows.append({
            "scenario_index": idx,
            "scenario_id": scenario["id"],
            "family": scenario["family"],
            "force_w_n": scenario["force_w_n"],
            "release": bool(scenario["release"]),
            "horizontal_angle_deg": scenario.get("horizontal_angle_deg"),
            "nominal_magnitude_n": scenario.get("nominal_magnitude_n"),
            "terminated": bool(terminated[mask].any()),
            "min_height_m": float(rr[:, 16].min()),
            "max_roll_abs_deg": float(np.abs(roll).max()),
            "max_pitch_abs_deg": float(np.abs(pitch).max()),
            "force_rmse_n": float(np.sqrt(np.mean(force_err**2))),
            "confidence_force_mean": float(rr[:, 13].mean()),
        })
    return {
        "seed": int(manifest["seed"]),
        "candidate": manifest["candidate"],
        "samples": int(len(r)),
        "scenarios": len(rows),
        "terminated_scenarios": int(sum(x["terminated"] for x in rows)),
        "completed_scenarios": int(sum(not x["terminated"] for x in rows)),
        "rows": rows,
        "manifest": manifest,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    runs = []
    for category in ("nominal", "robust"):
        for run in sorted((root / category).glob("seed*")):
            if list(run.glob("manifest_seed*.json")):
                item = one(run)
                item["category"] = category
                runs.append(item)
    groups = {}
    for category in ("nominal", "robust"):
        items = [x for x in runs if x["category"] == category]
        total = sum(x["scenarios"] for x in items)
        completed = sum(x["completed_scenarios"] for x in items)
        rows = [r for x in items for r in x["rows"]]
        by_family = {}
        for family in ("horizontal", "vertical", "combined"):
            sub = [r for r in rows if r["family"] == family]
            by_family[family] = {
                "total": len(sub),
                "completed": sum(not r["terminated"] for r in sub),
                "completion_rate": (sum(not r["terminated"] for r in sub) / len(sub)) if sub else None,
            }
        by_angle = {}
        for angle in sorted({r["horizontal_angle_deg"] for r in rows if r["horizontal_angle_deg"] is not None}):
            sub = [r for r in rows if r["horizontal_angle_deg"] == angle]
            by_angle[str(angle)] = {"total": len(sub), "completed": sum(not r["terminated"] for r in sub)}
        by_release = {}
        for release in (False, True):
            sub = [r for r in rows if r["release"] is release]
            by_release["release" if release else "hold"] = {
                "total": len(sub),
                "completed": sum(not r["terminated"] for r in sub),
                "completion_rate": sum(not r["terminated"] for r in sub) / len(sub) if sub else None,
            }
        by_horizontal_force = {}
        horizontal_levels = sorted({round(float(np.linalg.norm(r["force_w_n"][:2])), 3) for r in rows})
        for level in horizontal_levels:
            sub = [r for r in rows if round(float(np.linalg.norm(r["force_w_n"][:2])), 3) == level]
            by_horizontal_force[str(level)] = {
                "total": len(sub), "completed": sum(not r["terminated"] for r in sub),
                "completion_rate": sum(not r["terminated"] for r in sub) / len(sub) if sub else None,
            }
        by_vertical_force = {}
        vertical_levels = sorted({abs(float(r["force_w_n"][2])) for r in rows})
        for level in vertical_levels:
            sub = [r for r in rows if abs(abs(float(r["force_w_n"][2])) - level) < 1e-6]
            by_vertical_force[str(level)] = {
                "total": len(sub), "completed": sum(not r["terminated"] for r in sub),
                "completion_rate": sum(not r["terminated"] for r in sub) / len(sub) if sub else None,
            }
        groups[category] = {
            "runs": len(items), "total_scenarios": total,
            "completed_scenarios": completed,
            "stable_completion_rate": completed / total if total else 0.0,
            "by_family": by_family, "by_horizontal_angle": by_angle,
            "by_hold_release": by_release,
            "by_horizontal_force_n": by_horizontal_force,
            "by_vertical_force_abs_n": by_vertical_force,
        }
    result = {
        "schema_version": "gate41-direction-matrix-certification-v1",
        "acceptance": {"stable_completion_rate_min": 0.95, "scope": "MJLab/MuJoCo only; no real robot"},
        "summary": groups,
        "runs": runs,
        "status": "PASSED" if all(groups.get(c, {}).get("stable_completion_rate", 0) >= 0.95 for c in ("nominal", "robust")) else "FAILED",
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
