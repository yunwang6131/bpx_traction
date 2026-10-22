#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch, onnxruntime as ort
from world_force.model import MissingStateTCN


class Normalized(torch.nn.Module):
    def __init__(self,m,mean,std): super().__init__(); self.m=m; self.register_buffer("mean",mean); self.register_buffer("std",std)
    def forward(self,x): return self.m((x-self.mean)/self.std)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",required=True); ap.add_argument("--dataset",required=True); ap.add_argument("--output",required=True); a=ap.parse_args(); md=Path(a.model_dir)
    ck=torch.load(md/"estimator.pt",map_location="cpu"); m=MissingStateTCN(input_dim=ck.get("input_dim",len(ck["mean"]))).eval(); m.load_state_dict(ck["model"]); pt=Normalized(m,ck["mean"],ck["std"]).eval(); x=torch.from_numpy(np.load(Path(a.dataset)/"test.npz")["sensor_history"][:128])
    with torch.no_grad(): pv,pc=pt(x)
    sess=ort.InferenceSession(str(md/"estimator.onnx"),providers=["CPUExecutionProvider"]); ov,oc=sess.run(None,{"sensor_history":x.numpy()})
    report={"values_max_abs_error":float(np.max(np.abs(pv.numpy()-ov))),"confidence_max_abs_error":float(np.max(np.abs(pc.numpy()-oc))),"threshold":1e-4}; report["passed"]=max(report["values_max_abs_error"],report["confidence_max_abs_error"])<report["threshold"]
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n"); raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__": main()
