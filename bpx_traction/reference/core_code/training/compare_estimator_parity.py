#!/usr/bin/env python3
"""Replay identical C++ history rows through Python ONNX and compare F_hat."""
from pathlib import Path
import argparse, json
import numpy as np
import onnxruntime as ort

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scene-root',required=True);ap.add_argument('--onnx',required=True);a=ap.parse_args();root=Path(a.scene_root);sess=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider']);name=sess.get_inputs()[0].name
 rows=[]
 for p in sorted(root.glob('*/controller_history.csv')):
  data=np.loadtxt(p,delimiter=',',ndmin=2)
  if data.shape[1] != 1+25*82+3: raise RuntimeError(f'{p}: unexpected columns {data.shape[1]}')
  x=data[:,1:1+25*82].reshape(-1,25,82).astype(np.float32); cpp=data[:,-3:].astype(np.float32)
  py=np.concatenate([sess.run(None,{name:x[i:i+1]})[0] for i in range(len(x))],axis=0); d=np.abs(py-cpp)
  rows.append({'scene_id':p.parent.name,'frames':len(data),'max_abs_n':float(d.max()*80.0),'mean_abs_n':float(d.mean()*80.0),'rmse_n':float(np.sqrt(np.mean(d*d))*80.0),'input_finite':bool(np.isfinite(x).all()),'history_shape':[25,82]})
 out={'scene_count':len(rows),'max_abs_n':max(r['max_abs_n'] for r in rows),'mean_abs_n':float(np.mean([r['mean_abs_n'] for r in rows])),'scenes':rows,'contract':'same recorded 25x82 tensor -> Python onnxruntime vs C++ Ort output'}
 (root/'estimator_parity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
