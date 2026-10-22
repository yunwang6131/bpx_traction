#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from src.tasks.velocity.rl.force_estimator import ForceEstimator

def load(p):
 d=np.load(p); return torch.from_numpy(d['sensor_history']).float(),torch.from_numpy(d['privileged_targets'][:,12:15]).float()/80.
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--static',required=True);ap.add_argument('--jump',required=True);ap.add_argument('--init',required=True);ap.add_argument('--output',required=True);ap.add_argument('--epochs',type=int,default=30);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True); init=torch.load(a.init,map_location='cpu',weights_only=False)
 sx,sy=load(Path(a.static)/'train.npz');jx,jy=load(Path(a.jump)/'jump.npz')
 # Preserve the original behavior by presenting static windows three times.
 tx=torch.cat((sx,sx,sx,jx));ty=torch.cat((sy,sy,sy,jy))
 vx,vy=load(Path(a.static)/'validation.npz');jvx,jvy=load(Path(a.jump)/'validation.npz') if (Path(a.jump)/'validation.npz').exists() else (jx[:len(jx)//6],jy[:len(jy)//6]);vx=torch.cat((vx,jvx));vy=torch.cat((vy,jvy))
 mean=init['mean'];std=init['std'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');m=ForceEstimator(82,25,init['hidden_dim']).to(device);m.load_state_dict(init['model']);opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-5);loader=DataLoader(TensorDataset(tx,ty),batch_size=512,shuffle=True);best=1e9;hist=[]
 for ep in range(a.epochs):
  m.train();total=0
  for x,y in loader:
   p=m(((x-mean)/std).to(device));y=y.to(device);active=y.norm(dim=1)>0.025;per=torch.nn.functional.smooth_l1_loss(p,y,reduction='none').mean(1);loss=(per*torch.where(active,torch.full_like(per,4.),torch.ones_like(per))).mean()
   if active.any():loss=loss+0.15*(1-torch.nn.functional.cosine_similarity(p[active],y[active],dim=1).mean())
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();total+=float(loss)*len(x)
  m.eval();
  with torch.no_grad(): val=torch.nn.functional.smooth_l1_loss(m(((vx-mean)/std).to(device)),vy.to(device)).item()
  hist.append({'epoch':ep+1,'train_loss':total/len(tx),'validation_loss':val})
  if val<best:best=val;torch.save({'model':m.state_dict(),'mean':mean,'std':std,'input_dim':82,'history_length':25,'hidden_dim':init['hidden_dim']},out/'estimator.pt')
 ck=torch.load(out/'estimator.pt',map_location='cpu',weights_only=False);m=ForceEstimator(82,25,ck['hidden_dim']).eval();m.load_state_dict(ck['model'])
 class W(torch.nn.Module):
  def __init__(self,m,mean,std):super().__init__();self.m=m;self.register_buffer('mean',mean);self.register_buffer('std',std)
  def forward(self,x):return self.m((x-self.mean)/self.std)
 w=W(m,ck['mean'],ck['std']).eval();torch.onnx.export(w,torch.zeros(1,25,82),out/'force_estimator.onnx',input_names=['proprio_history'],output_names=['estimated_force'],opset_version=18,dynamo=False);torch.jit.trace(w,torch.zeros(1,25,82)).save(str(out/'force_estimator.ts'));np.savez(out/'normalization.npz',mean=ck['mean'].numpy(),std=ck['std'].numpy());(out/'training_report.json').write_text(json.dumps({'best_validation_loss':best,'epochs':hist,'input_shape':[1,25,82],'output_shape':[1,3],'initialized_from':a.init},indent=2)+'\n')
if __name__=='__main__':main()
