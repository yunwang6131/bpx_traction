#!/usr/bin/env python3
from pathlib import Path
import argparse, numpy as np, json
from collect_sim_estimator_dataset import collect

def main():
 import mjlab.tasks, src.tasks  # noqa: F401
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--target',choices=['east_hold','dir_mag_jump'],required=True);ap.add_argument('--device',default='cuda:1');ap.add_argument('--num-envs',type=int,default=128);ap.add_argument('--steps',type=int,default=500);a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 x,y,e=collect(root,seed=43000+(0 if a.target=='east_hold' else 1),num_envs=a.num_envs,steps=a.steps,stride=5,device=a.device,jump=(a.target=='dir_mag_jump'),target=a.target)
 rng=np.random.default_rng(43025);idx=rng.permutation(len(x));cuts=[0,int(.7*len(x)),int(.85*len(x)),len(x)];sizes={}
 for n,l,h in zip(['train','validation','test'],cuts[:-1],cuts[1:]):
  j=idx[l:h];np.savez_compressed(out/(n+'.npz'),sensor_history=x[j],privileged_targets=y[j]);sizes[n]=len(j)
 (out/'manifest.json').write_text(json.dumps({'schema_version':'gate41-targeted-v1','input_shape':[25,82],'force_label_frame':'body','target':a.target,'sizes':sizes},indent=2)+'\n');print(sizes)
if __name__=='__main__':main()
