from pathlib import Path
import numpy as np,json,hashlib
ROOT=Path(__file__).parent

def stats(x):
 e=x['pred'].astype(np.float64)-x['true'].astype(np.float64);z=e[:,2]
 return dict(n=len(z),trajectories=len(np.unique(x['unit'])),bias=float(z.mean()),mae=float(abs(z).mean()),rmse=float(np.sqrt((z*z).mean())),p95=float(np.percentile(abs(z),95)),true_mean=float(x['true'][:,2].mean()),pred_mean=float(x['pred'][:,2].mean()),sse=float((z*z).sum())) if len(z) else dict(n=0)
def select(x,m):return {k:v[m] for k,v in x.items()}
def bins(x,key,edges):
 return {f'{a:g}–{b:g}':stats(select(x,(x[key]>=a)&(x[key]<b))) for a,b in zip(edges[:-1],edges[1:])}
def groups(x,key):return {str(a):stats(select(x,x[key]==a)) for a in np.unique(x[key])}
def merge(xs):return {k:np.concatenate([x[k] for x in xs]) for k in xs[0]}
res={'models':{},'files':{}}
for kind in ['latest25','delta25']:
 full=[];sparse=[]
 for si,scenario in enumerate(['ramp','release','reverse','sweep','full3d']):
  f=next((ROOT/'data').glob(f'{scenario}_{kind}_*.npz'));d=np.load(f);p=d['prediction']*80;y=d['labels']*80;t,n,_=p.shape
  res['files'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
  time,env=np.indices((t,n));mag=np.linalg.norm(y,axis=-1);cos=np.divide(y[...,2],mag,out=np.zeros_like(mag),where=mag>1e-5)
  direction=np.where(mag<2,'near_zero',np.where(cos>.5,'up',np.where(cos<-.5,'down','horizontal')))
  # Index of first label after command applied at 50,400,600 is 51,401,601.
  transient=np.zeros((t,n),bool)
  for event in ([51,401] if scenario=='release' else [51,601] if scenario=='reverse' else [51]):transient|=(time>=event)&(time<event+50)
  reset=np.zeros((t,n),bool)
  for k,j in zip(*np.where(d['dones'])):reset[max(0,k-25):min(t,k+51),j]=True
  data=dict(pred=p.reshape(-1,3),true=y.reshape(-1,3),mag=mag.ravel(),cos=cos.ravel(),direction=direction.ravel(),scene=np.full(t*n,scenario),time=time.ravel(),unit=(env+si*n).ravel(),transient=transient.ravel(),reset=reset.ravel())
  full.append(select(data,time.ravel()>=50))
  hs=d['history_samples']; indices=[];frames=[]
  for i in range(len(hs)):
   for j in range(25):
    k=i*100-j
    if k<50:continue
    h=hs[i,:,j];ok=np.linalg.norm(h[:,9:12],axis=1)>.5
    indices.extend((k*n+np.where(ok)[0]).tolist());frames.extend(h[ok])
  h=np.asarray(frames);sp={k:v[np.asarray(indices)] for k,v in data.items()};g=h[:,9:12];gn=np.linalg.norm(g,axis=1)
  sp['tilt']=np.degrees(np.arccos(np.clip(-g[:,2]/gn,-1,1)));sp['roll']=np.degrees(np.arctan2(-g[:,1],-g[:,2]));sp['pitch']=np.degrees(np.arcsin(np.clip(g[:,0]/gn,-1,1)));sp['height']=h[:,12];sp['contact']=np.sum(h[:,54:58]>.5,axis=1)
  sp['gravity_norm']=gn;sparse.append(sp)
 x=merge(full);s=merge(sparse);np.savez_compressed(ROOT/f'{kind}_pose_samples.npz',**s)
 r={'all':stats(x),'magnitude':bins(x,'mag',[0,5,15.0001,30.0001,45.001]),'direction':groups(x,'direction'),'scene':groups(x,'scene'),'transient':groups(x,'transient'),'reset':groups(x,'reset'),'signed_fz':bins(dict(x,fz=x['true'][:,2]),'fz',[-46,-30,-15,-5,5,15,30,46]),'pose_all':stats(s),'tilt':bins(s,'tilt',[0,1,2,5,10,20,40,181]),'height':bins(s,'height',[0,.2,.25,.3,.35,2]),'contacts':groups(s,'contact'),'gravity_norm_range':[float(s['gravity_norm'].min()),float(s['gravity_norm'].max())]}
 r['cross']={};r['pose_cross']={}
 for a,b in [(0,15.0001),(15.0001,30.0001),(30.0001,45.001)]:
  for direction in ['up','horizontal','down']:
   m=(x['mag']>=a)&(x['mag']<b)&(x['direction']==direction);key=f'{a:g}-{b:g}/{direction}';r['cross'][key]=stats(select(x,m))
   m=(s['mag']>=a)&(s['mag']<b)&(s['direction']==direction);r['pose_cross'][key]=bins(select(s,m),'tilt',[0,1,2,5,10,20,40,181])
 r['scene_direction']={scene:groups(select(x,x['scene']==scene),'direction') for scene in np.unique(x['scene'])}
 # A descriptive linear slope is not a proposed calibration or causal estimate.
 r['fz_linear_fit']={}
 for name,mask in [('all',np.ones(len(x['mag']),bool)),('up',x['direction']=='up'),('down',x['direction']=='down')]:
  xx=x['true'][mask,2];yy=x['pred'][mask,2];coef=np.polyfit(xx,yy,1);r['fz_linear_fit'][name]={'slope':float(coef[0]),'intercept':float(coef[1])}
 res['models'][kind]=r
json.dump(res,open(ROOT/'diagnosis.json','w'),indent=2,ensure_ascii=False)
for k,r in res['models'].items():
 print(k,json.dumps({a:r[a] for a in ['all','magnitude','direction','tilt','reset','fz_linear_fit']},ensure_ascii=False,indent=2))
