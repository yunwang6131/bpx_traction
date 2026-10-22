#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--directory",required=True);ap.add_argument("--output",required=True);a=ap.parse_args();d=Path(a.directory);runs=[]
    for mf in sorted(d.glob("manifest_seed*.json")):
        m=json.loads(mf.read_text());x=np.load(d/m["file"])["estimator_in_loop_rollout"];active=np.linalg.norm(x[:,4:7],axis=1)>2;cos=np.sum(x[active,4:7]*x[active,7:10],1)/(np.linalg.norm(x[active,4:7],axis=1)*np.linalg.norm(x[active,7:10],axis=1)+1e-9)
        runs.append({"seed":m["seed"],"samples":len(x),"terminated_samples":int(x[:,3].sum()),"stable_completion":bool(x[:,3].sum()==0),"min_height_m":float(x[:,16].min()),"mean_speed_mps":float(np.linalg.norm(x[:,17:20],axis=1).mean()),"force_rmse_n":float(np.sqrt(np.mean((x[:,7:10]-x[:,4:7])**2))),"force_direction_mean_deg":float(np.degrees(np.arccos(np.clip(cos,-1,1))).mean()),"force_confidence_mean":float(x[:,13].mean())})
    report={"evidence":"actual estimator-in-loop MJLab/MuJoCo","group":"C","gate_result":"fail" if not runs or any(not r["stable_completion"] for r in runs) else "pass","runs":runs,"decision":"Do not fine-tune Actor and do not proceed to real robot while G3/G4 fail."};Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
