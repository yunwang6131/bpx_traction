#!/usr/bin/env python3
"""Conservative force-estimator fine-tune with a frozen baseline teacher."""
from pathlib import Path
import argparse, json
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from src.tasks.velocity.rl.force_estimator import ForceEstimator

def load(p):
 d=np.load(p); return torch.from_numpy(d['sensor_history']).float(), torch.from_numpy(d['privileged_targets'][:,12:15]).float()/80.0

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--static',required=True); ap.add_argument('--jump',required=True); ap.add_argument('--init',required=True); ap.add_argument('--output',required=True); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--lr',type=float,default=5e-5); a=ap.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 ck=torch.load(a.init,map_location='cpu',weights_only=False); sx,sy=load(Path(a.static)/'train.npz'); jx,jy=load(Path(a.jump)/'jump.npz'); vx,vy=load(Path(a.static)/'validation.npz')
 # Keep the stable distribution present in every epoch; the jump cohort is
 # sampled separately so it cannot erase the normal-force behavior.
 static_loader=DataLoader(TensorDataset(sx,sy),batch_size=512,shuffle=True,drop_last=True); jump_loader=DataLoader(TensorDataset(jx,jy),batch_size=512,shuffle=True,drop_last=True)
 mean,std=ck['mean'],ck['std']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 teacher=ForceEstimator(82,25,ck['hidden_dim']).eval(); teacher.load_state_dict(ck['model']);
 for p in teacher.parameters(): p.requires_grad_(False)
 student=ForceEstimator(82,25,ck['hidden_dim']).to(device); student.load_state_dict(ck['model']); teacher=teacher.to(device)
 opt=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=1e-5);best=1e9; hist=[]
 for ep in range(a.epochs):
  student.train(); total=0.; n=0; js=iter(jump_loader)
  for xs,ys in static_loader:
   try: xj,yj=next(js)
   except StopIteration: js=iter(jump_loader);xj,yj=next(js)
   xs,ys,xj,yj=[z.to(device) for z in (xs,ys,xj,yj)]
   ps=student((xs-mean.to(device))/std.to(device)); pj=student((xj-mean.to(device))/std.to(device))
   with torch.no_grad(): ts=teacher((xs-mean.to(device))/std.to(device)); tj=teacher((xj-mean.to(device))/std.to(device))
   static_sup=torch.nn.functional.smooth_l1_loss(ps,ys); static_dist=torch.nn.functional.mse_loss(ps,ts)
   active=yj.norm(dim=1)>0.025; jump_per=torch.nn.functional.smooth_l1_loss(pj,yj,reduction='none').mean(1); jump_sup=(jump_per*torch.where(active,torch.full_like(jump_per,5.),torch.ones_like(jump_per))).mean(); jump_dist=0.1*torch.nn.functional.mse_loss(pj,tj)
   loss=static_sup+3.0*static_dist+jump_sup+jump_dist
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),0.5);opt.step();total+=float(loss)*(len(xs));n+=len(xs)
  student.eval();
  with torch.no_grad():
   sv=torch.nn.functional.smooth_l1_loss(student((vx.to(device)-mean.to(device))/std.to(device)),vy.to(device)).item()
   # Use a deterministic jump validation slice for model selection.
   jv_x,jv_y=jx[:min(4096,len(jx))],jy[:min(4096,len(jy))]
   jv=torch.nn.functional.smooth_l1_loss(student((jv_x.to(device)-mean.to(device))/std.to(device)),jv_y.to(device)).item()
  score=sv+0.35*jv;hist.append({'epoch':ep+1,'train_loss':total/max(1,n),'static_val':sv,'jump_val':jv,'score':score})
  if score<best:best=score;torch.save({'model':student.state_dict(),'mean':mean,'std':std,'input_dim':82,'history_length':25,'hidden_dim':ck['hidden_dim']},out/'estimator.pt')
 bestck=torch.load(out/'estimator.pt',map_location='cpu',weights_only=False);m=ForceEstimator(82,25,bestck['hidden_dim']).eval();m.load_state_dict(bestck['model'])
 class W(torch.nn.Module):
  def __init__(self,m,mean,std):super().__init__();self.m=m;self.register_buffer('mean',mean);self.register_buffer('std',std)
  def forward(self,x):return self.m((x-self.mean)/self.std)
 w=W(m,bestck['mean'],bestck['std']).eval();torch.onnx.export(w,torch.zeros(1,25,82),out/'force_estimator.onnx',input_names=['proprio_history'],output_names=['estimated_force'],opset_version=18,dynamo=False);torch.jit.trace(w,torch.zeros(1,25,82)).save(str(out/'force_estimator.ts'));np.savez(out/'normalization.npz',mean=bestck['mean'].numpy(),std=bestck['std'].numpy());(out/'training_report.json').write_text(json.dumps({'best_score':best,'epochs':hist,'input_shape':[1,25,82],'output_shape':[1,3],'teacher':a.init,'static_distillation_weight':3.0,'jump_supervision_weight':1.0},indent=2)+'\n')
if __name__=='__main__':main()
