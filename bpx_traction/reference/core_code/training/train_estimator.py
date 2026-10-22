#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from world_force.model import MissingStateTCN


def load(path: Path):
    d=np.load(path); return torch.from_numpy(d["sensor_history"]), torch.from_numpy(d["privileged_targets"])


def task_loss(pred, target, confidence, velocity_weight=1.0, height_weight=8.0):
    vel=nn.functional.smooth_l1_loss(pred[:,:3],target[:,:3])
    height=nn.functional.smooth_l1_loss(pred[:,3:4],target[:,3:4])
    contact=nn.functional.binary_cross_entropy(pred[:,4:8],target[:,4:8])
    load=nn.functional.smooth_l1_loss(pred[:,8:12],target[:,8:12])
    active=torch.linalg.norm(target[:,12:15],dim=1)>2
    force_per=nn.functional.smooth_l1_loss(pred[:,12:15]/40.0,target[:,12:15]/40.0,reduction="none").mean(1)
    force=(force_per*torch.where(active,torch.full_like(force_per,4.0),torch.ones_like(force_per))).mean()
    direction=1-torch.nn.functional.cosine_similarity(pred[:,12:15],target[:,12:15],dim=1,eps=1e-6)
    direction=direction[active].mean() if active.any() else direction.mean()*0
    errors=torch.stack(((pred[:,:3]-target[:,:3]).abs().mean(1),(pred[:,3]-target[:,3]).abs(),(pred[:,4:8]-target[:,4:8]).abs().mean(1),((pred[:,12:15]-target[:,12:15]).abs()/40).mean(1)),1).detach()
    confidence_target=torch.exp(-3*errors).clamp(0,1)
    calibration=nn.functional.mse_loss(confidence,confidence_target)
    return velocity_weight*vel+height_weight*height+contact+load+3*force+1.5*direction+.5*calibration


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",required=True); ap.add_argument("--output",required=True); ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--velocity-weight",type=float,default=1.0); ap.add_argument("--height-weight",type=float,default=8.0); ap.add_argument("--channels",type=int,default=64); a=ap.parse_args()
    data=Path(a.dataset); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tx,ty=load(data/"train.npz"); vx,vy=load(data/"validation.npz")
    mean=tx.mean((0,1)); std=tx.std((0,1)).clamp_min(1e-4)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim=tx.shape[-1]
    model=MissingStateTCN(input_dim=input_dim,channels=a.channels).to(device); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=512,shuffle=True)
    best=float("inf"); history=[]
    for epoch in range(a.epochs):
        model.train(); total=0
        for x,y in loader:
            x=((x-mean)/std).to(device); y=y.to(device); p,c=model(x); loss=task_loss(p,y,c,a.velocity_weight,a.height_weight)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=loss.item()*len(x)
        model.eval()
        with torch.no_grad(): p,c=model(((vx-mean)/std).to(device)); valid=task_loss(p,vy.to(device),c,a.velocity_weight,a.height_weight).item()
        history.append({"epoch":epoch+1,"train_loss":total/len(tx),"validation_loss":valid})
        if valid<best: best=valid; torch.save({"model":model.state_dict(),"mean":mean,"std":std,"input_dim":input_dim,"channels":a.channels},out/"estimator.pt")
    ckpt=torch.load(out/"estimator.pt",map_location="cpu"); model=MissingStateTCN(input_dim=ckpt["input_dim"],channels=ckpt.get("channels",64)).eval(); model.load_state_dict(ckpt["model"])
    wrapper=Normalized(model,ckpt["mean"],ckpt["std"]).eval(); example=torch.zeros(1,25,ckpt["input_dim"])
    torch.onnx.export(wrapper,example,out/"estimator.onnx",input_names=["sensor_history"],output_names=["values","confidence"],dynamic_axes={"sensor_history":{0:"batch"},"values":{0:"batch"},"confidence":{0:"batch"}},opset_version=17)
    torch.jit.trace(wrapper,example).save(str(out/"estimator.ts"))
    np.savez(out/"normalization.npz",mean=ckpt["mean"].numpy(),std=ckpt["std"].numpy())
    (out/"training_report.json").write_text(json.dumps({"device":str(device),"best_validation_loss":best,"epochs":history},indent=2)+"\n")


class Normalized(nn.Module):
    def __init__(self,model,mean,std): super().__init__(); self.model=model; self.register_buffer("mean",mean); self.register_buffer("std",std)
    def forward(self,x): return self.model((x-self.mean)/self.std)


if __name__=="__main__": main()
