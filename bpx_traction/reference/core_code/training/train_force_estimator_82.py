#!/usr/bin/env python3
"""Train the deployable 82-D temporal force estimator from MuJoCo windows."""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.tasks.velocity.rl.force_estimator import ForceEstimator

def load(p):
    d=np.load(p); return torch.from_numpy(d['sensor_history']).float(), torch.from_numpy(d['privileged_targets'][:,12:15]).float()/80.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--hidden',type=int,default=64); a=ap.parse_args()
    ds=Path(a.dataset); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tx,ty=load(ds/'train.npz'); vx,vy=load(ds/'validation.npz')
    mean=tx.mean((0,1)); std=tx.std((0,1)).clamp_min(1e-4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ForceEstimator(82,25,hidden_dim=a.hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(tx,ty),batch_size=512,shuffle=True)
    best=1e9; hist=[]
    for ep in range(a.epochs):
        model.train(); total=0.
        for x,y in loader:
            p=model(((x-mean)/std).to(device)); y=y.to(device)
            active=(y.norm(dim=1)>0.025); per=torch.nn.functional.smooth_l1_loss(p,y,reduction='none').mean(1)
            loss=(per*torch.where(active,torch.full_like(per,4.),torch.ones_like(per))).mean()
            loss=loss+0.25*(1-torch.nn.functional.cosine_similarity(p[active],y[active],dim=1).mean()) if active.any() else loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total+=float(loss)*len(x)
        model.eval()
        with torch.no_grad(): val=torch.nn.functional.smooth_l1_loss(model(((vx-mean)/std).to(device)),vy.to(device)).item()
        hist.append({'epoch':ep+1,'train_loss':total/len(tx),'validation_loss':val})
        if val<best: best=val; torch.save({'model':model.state_dict(),'mean':mean,'std':std,'input_dim':82,'history_length':25,'hidden_dim':a.hidden},out/'estimator.pt')
    ck=torch.load(out/'estimator.pt',map_location='cpu'); model=ForceEstimator(82,25,a.hidden).eval(); model.load_state_dict(ck['model'])
    class Wrapped(torch.nn.Module):
        def __init__(self,m,mean,std): super().__init__(); self.m=m; self.register_buffer('mean',mean); self.register_buffer('std',std)
        def forward(self,x): return self.m((x-self.mean)/self.std)
    w=Wrapped(model,ck['mean'],ck['std']).eval(); torch.onnx.export(w,torch.zeros(1,25,82),out/'force_estimator.onnx',input_names=['proprio_history'],output_names=['estimated_force'],dynamic_axes={'proprio_history':{0:'batch'},'estimated_force':{0:'batch'}},opset_version=18,dynamo=False)
    torch.jit.trace(w,torch.zeros(1,25,82)).save(str(out/'force_estimator.ts')); np.savez(out/'normalization.npz',mean=ck['mean'].numpy(),std=ck['std'].numpy()); (out/'training_report.json').write_text(json.dumps({'device':str(device),'best_validation_loss':best,'epochs':hist,'input_shape':[1,25,82],'output_shape':[1,3]},indent=2)+'\n')
if __name__=='__main__': main()
