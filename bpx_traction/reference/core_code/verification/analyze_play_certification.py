#!/usr/bin/env python3
"""Audit a complete v15 estimator-in-loop Play certification matrix."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


SCENARIOS = ["zero_stand", "plus_x_heading", "plus_x_yaw90", "lateral_vertical", "diagonal", "release"]


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def quat_roll_pitch(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    return np.rad2deg(roll),np.rad2deg(pitch)


def run_metrics(npz_path: Path, manifest_path: Path, category: str):
    m=json.loads(manifest_path.read_text()); z=np.load(npz_path); r=z["estimator_in_loop_rollout"]; d=z["extended_diagnostics"]
    cols={n:i for i,n in enumerate(m["columns"])}; scene_col=cols["scenario_index"]
    frame_map={int(x["scenario_index"]):x["scenario_id"] for x in m["scenario_frames"]}
    term_scenes=sorted({int(x) for x in r[r[:,cols["terminated"]]>0,scene_col]})
    true_f=r[:,4:7]; pred_f=r[:,7:10]; active=np.linalg.norm(true_f,axis=1)>2
    cos=np.sum(true_f[active]*pred_f[active],axis=1)/(np.linalg.norm(true_f[active],axis=1)*np.linalg.norm(pred_f[active],axis=1)+1e-9)
    q=d[:,:4]; roll,pitch=quat_roll_pitch(q); torque=d[:,31:43]
    release_idx=np.where((r[:,scene_col]==5)&(np.linalg.norm(true_f,axis=1)<=2))[0]
    release_delay=None; release_min=None; release_end=None; release_last_1s_mean=None; release_sustained=False
    if len(release_idx):
        start=release_idx[0]; post=np.linalg.norm(pred_f[start:],axis=1); hit=np.where(post<=2)[0]
        if len(hit): release_delay=float(r[start+hit[0],cols["time_s"]]-r[start,cols["time_s"]])
        release_min=float(post.min()); release_end=float(post[-1]); release_last_1s_mean=float(post[-50:].mean()); release_sustained=bool(np.all(post[-25:]<=2))
    return {
      "category":category,"seed":int(m["seed"]),"file":str(npz_path),"file_sha256":sha256(npz_path),
      "samples":int(len(r)),"scenarios":len(frame_map),"scenario_ids":[frame_map[i] for i in sorted(frame_map)],
      "terminated_samples":int((r[:,cols["terminated"]]>0).sum()),"terminated_scenarios":[frame_map[i] for i in term_scenes],
      "completed_scenarios":len(frame_map)-len(term_scenes),"min_height_m":float(r[:,cols["pos_z"]].min()),
      "force_rmse_n":float(np.sqrt(np.mean((pred_f-true_f)**2))),"force_direction_mean_deg":float(np.degrees(np.arccos(np.clip(cos,-1,1))).mean()),
      "release_delay_to_2n_s":release_delay,"release_force_min_n":release_min,"release_force_end_n":release_end,"release_force_last_1s_mean_n":release_last_1s_mean,"release_sustained_below_2n_final_0p5s":release_sustained,"roll_abs_peak_deg":float(np.abs(roll).max()),"pitch_abs_peak_deg":float(np.abs(pitch).max()),
      "joint_torque_abs_peak_nm":float(np.abs(torque).max()),"oracle_force":bool(m["diagnostic_oracle_force"]),"oracle_state":bool(m["diagnostic_oracle_state"]),
      "input_privileged_leakage":bool(m["input_privileged_leakage"]),"perturbations":m["perturbations"]
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",required=True);ap.add_argument("--cpu-report",required=True);ap.add_argument("--onnx-report",required=True);ap.add_argument("--output",required=True);a=ap.parse_args()
    root=Path(a.root); runs=[]
    for category in ("nominal","robust"):
        for manifest in sorted((root/category).glob("seed*/manifest_seed*.json")):
            m=json.loads(manifest.read_text()); runs.append(run_metrics(manifest.parent/m["file"],manifest,category))
    by={c:[x for x in runs if x["category"]==c] for c in ("nominal","robust")}
    summary={}
    for c,items in by.items():
        completed=sum(x["completed_scenarios"] for x in items); total=sum(x["scenarios"] for x in items)
        fails={s:sum(s in x["terminated_scenarios"] for x in items) for s in SCENARIOS}
        summary[c]={"runs":len(items),"completed_scenarios":completed,"total_scenarios":total,"stable_completion_rate":completed/total if total else 0,"failure_count_by_scenario":fails,"systematic_failure":any(v>=2 for v in fails.values())}
    cpu=json.loads(Path(a.cpu_report).read_text()); onnx=json.loads(Path(a.onnx_report).read_text())["onnx_consistency"]
    integrity=(len(runs)==8 and all(x["samples"]==2400 and x["scenarios"]==6 and x["scenario_ids"]==SCENARIOS and not x["oracle_force"] and not x["oracle_state"] and not x["input_privileged_leakage"] for x in runs))
    expected_robust={"sensor_noise":0.02,"sensor_bias":0.01,"random_walk_per_step":0.0003,"yaw_drift_deg_per_scenario":3.0,"dropout_probability":0.03,"delay_steps":2}
    robust_exact=all(all(abs(float(x["perturbations"][k])-float(v))<1e-12 for k,v in expected_robust.items()) for x in by["robust"])
    runtime_p99=max(cpu["test"]["inference_ms_p99"],cpu["ood"]["inference_ms_p99"])
    criteria={"data_integrity_and_no_oracle":integrity,"nominal_rate_at_least_95pct":summary["nominal"]["stable_completion_rate"]>=.95,"robust_rate_at_least_95pct":summary["robust"]["stable_completion_rate"]>=.95,"no_systematic_scenario_failure":not summary["nominal"]["systematic_failure"] and not summary["robust"]["systematic_failure"],"robust_injections_exact":robust_exact,"onnx_consistency_below_1e_4":bool(onnx["passed"]),"cpu_p99_below_6ms":runtime_p99<6.0}
    stable_pass=all(criteria.values()); release_all_settled=all(x["release_sustained_below_2n_final_0p5s"] for x in runs)
    status="PLAY_CERTIFICATION_FAILED" if not stable_pass else ("PLAY_CERTIFICATION_PASSED" if release_all_settled else "PLAY_CERTIFICATION_CONDITIONAL")
    out={"schema_version":"gate41-play-certification-v1","candidate":"missing_state_world_v15_release","policy":"model_17488 frozen actor","seeds":{"nominal":[52001,52002,52003,52004],"robust":[52005,52006,52007,52008]},"scenario_order":SCENARIOS,"acceptance":{"completion_rate_min":.95,"cpu_p99_limit_ms":6.0,"onnx_max_abs_error":1e-4,"release_delay_absolute_threshold":"not specified by scheme; report only","release_diagnostic":"2 N threshold; final 0.5 s sustained check is an explicit open-risk diagnostic, not a scheme-defined pass threshold"},"summary":summary,"runtime_p99_ms":runtime_p99,"onnx_consistency":onnx,"criteria":criteria,"stability_gate":"PASSED" if stable_pass else "FAILED","release_all_runs_sustained_below_2n_final_0p5s":release_all_settled,"runs":runs,"status":status,"scope":"MJLab/MuJoCo estimator-in-loop simulation only; not Sim2Sim or real robot certification"}
    Path(a.output).write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")


if __name__=="__main__":main()
