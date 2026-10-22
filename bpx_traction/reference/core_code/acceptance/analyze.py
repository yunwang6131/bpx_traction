from pathlib import Path
import os,json,time,hashlib,traceback
import numpy as np,torch
from scipy.stats import t as student_t
from torch.utils.tensorboard import SummaryWriter
from direction_balance_common import Data
from src.tasks.velocity.rl.force_estimator import ForceEstimator
from acceptance_v2_metrics import score,masks,event_metrics
from mjlab.utils.torch import configure_torch_backends
O=Path('logs/rsl_rl/go2_velocity/acceptance_v2');R=Path('logs/acceptance_v2');TB=Path('logs/rsl_rl/go2_velocity/force_estimator_ft_20261014_v2/tensorboard')
configure_torch_backends(allow_tf32=False,deterministic=True);torch.set_num_threads(2)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(path,value):
 tmp=path.with_suffix('.tmp');json.dump(value,open(tmp,'w'),indent=2,allow_nan=False);tmp.replace(path)
def flatten(row):
 out={k:v for k,v in row.items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
 for k,suffix in [('component_mae','mae'),('component_rmse','rmse')]:
  for axis,v in zip(['fx','fy','fz'],row.get(k,[])):out[axis+'_'+suffix]=v
 for k,v in row.get('zero_false_positive_pct',{}).items():out['zero_false_positive_above_'+k+'_pct']=v
 return out
models={}
for name,path in [('selected','logs/rsl_rl/go2_velocity/convergence_v1/best.pt'),('start','logs/rsl_rl/go2_velocity/causal_v3_extended/aligned_window_720.pt')]:
 m=ForceEstimator(82,25,32).cpu().eval();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);models[name]=m
lock=json.load(open(O/'lock.json'));results={};done=set();(O/'runs').mkdir(exist_ok=True)
while len(done)<20:
 progress=False
 for row in lock['plan']:
  seed=row['seed']
  for name in ['start','selected']:
   key=f'{name}_{seed}';fp=R/f'{key}.npz';side=R/f'{key}.json';evp=R/f'{key}.events.json';cache=O/'runs'/f'{key}.json'
   if key in done:continue
   if cache.exists():results[key]=json.load(open(cache));done.add(key);continue
   if not all(p.exists() for p in [fp,side,evp]):continue
   try:
    meta=json.load(open(side));events=json.load(open(evp))
   except json.JSONDecodeError:continue # sidecar may still be flushing; raw collector writes these last
   assert meta['model_sha256']==lock['model_sha256'][name] and meta['policy_sha256']==lock['policy_sha256']
   assert str(meta['physical_cuda_device'])==str(row['gpu'])
   assert meta['physics_audit']['substeps']==24000 and meta['physics_audit']['max_wrench_error_N']<1e-6
   assert meta['max_history_overlap_error']==0
   d=Data(seed,fp);p=d.pred.astype(float)*80;y=d.y.astype(float)*80;mask=masks(y,d.done);groups={}
   for group,sel in mask.items():groups[group]=score(p[sel],y[sel])
   z=np.load(fp)
   with torch.no_grad():replay=np.stack([models[name](torch.from_numpy(h)).numpy() for h in z['history_samples']])
   replay_err=float(abs(replay-z['prediction_samples']).max());assert replay_err<1e-5
   term_counts={}
   for e in events:
    for k,v in e['terms'].items():
     if v:term_counts[k]=term_counts.get(k,0)+1
   event_result=event_metrics(p,y,d.done)
   entry={'seed':seed,'name':name,'gpu':row['gpu'],'order':row['order'],'groups':groups,'transients':event_result,'terminations':{'count':len(events),'per_environment_hour':len(events)/(6000*.02*112/3600),'unique_envs':len(set(e['env'] for e in events)),'causes':term_counts,'by_env':{str(e):sum(r['env']==e for r in events) for e in set(r['env'] for r in events)}},'audits':{'data':d.audit,'runtime':meta,'torch_cpu_online_replay_max_normalized':replay_err,'torch_cpu_online_replay_max_N':replay_err*80},'timing_contract_version':2,'raw_sha256':sha(fp),'force_plan_sha256':hashlib.sha256(d.applied.tobytes()).hexdigest()}
   write_json(cache,entry);results[key]=entry;done.add(key);progress=True
   wr=SummaryWriter(str(O/f'result_{key}'))
   for g,r in groups.items():
    for k,v in flatten(r).items():wr.add_scalar('acceptance/'+g+'/'+k,v,seed)
   for en,er in event_result.items():
    for k,v in er['summary'].items():
     if isinstance(v,(int,float)):wr.add_scalar('transient/'+en+'/'+k,v,seed)
     elif isinstance(v,dict):
      for kk,vv in v.items():
       if vv is not None:wr.add_scalar('transient/'+en+'/'+k+'/'+kk,vv,seed)
   wr.add_scalar('closed_loop/termination_count',len(events),seed);wr.close()
   link=TB/('acceptance_v2_'+key)
   if not link.exists():link.symlink_to((O/f'result_{key}').resolve(),target_is_directory=True)
   print('ANALYZED',key,len(done),flush=True)
   del d,z,p,y,replay
 write_json(O/'analysis_status.json',{'complete_runs':len(done),'planned_runs':20,'keys':sorted(done)})
 if len(done)<20:time.sleep(10)
# Force plan equality is checked pair by pair, not assumed from seed.
for seed in range(30101,30111):assert results[f'selected_{seed}']['force_plan_sha256']==results[f'start_{seed}']['force_plan_sha256']
# Mean of per-seed metrics, not pooled quantiles; independent unit is seed, n=10.
def stats(values,alpha=.05):
 a=np.array(values,dtype=float);n=len(a);mu=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else 0;se=sd/np.sqrt(n);q=float(student_t.ppf(1-alpha/2,n-1)) if n>1 else 0
 return {'n_seeds':n,'mean':mu,'sd':sd,'min':float(a.min()),'max':float(a.max()),'ci95':[mu-q*se,mu+q*se]}
summary={};paired={};groupnames=list(results['selected_30101']['groups'])
for name in ['start','selected']:
 summary[name]={}
 for g in groupnames:
  rows=[flatten(results[f'{name}_{s}']['groups'][g]) for s in range(30101,30111)]
  keys=set.intersection(*(set(r) for r in rows));summary[name][g]={k:stats([r[k] for r in rows]) for k in sorted(keys)}
for g in groupnames:
 sr=[flatten(results[f'selected_{s}']['groups'][g]) for s in range(30101,30111)];br=[flatten(results[f'start_{s}']['groups'][g]) for s in range(30101,30111)];keys=set.intersection(*(set(r) for r in sr+br))
 paired[g]={k:stats([r[k]-b[k] for r,b in zip(sr,br)]) for k in sorted(keys)}
limits={'fx_mae':1,'fy_mae':1,'fz_mae':1,'fx_rmse':1.5,'fy_rmse':1.5,'fz_rmse':1.5,'mag_mae':1.5,'mag_rmse':1.5,'angle_mean':5,'angle_p95':10}
primary=[s+'/'+b for s in ['ramp','release','reverse','sweep','full3d'] for b in ['low','high']];K=len(primary)*len(limits);bounds={}
for g in primary:
 rows=[results[f'selected_{s}']['groups'][g] for s in range(30101,30111)];tests={}
 for k,lim in limits.items():
  st=summary['selected'][g][k];delta=float(student_t.ppf(1-.05/(2*K),9))*st['sd']/np.sqrt(10);tests[k]={'limit':lim,'seed_mean':st['mean'],'simultaneous_lower':st['mean']-delta,'simultaneous_upper':st['mean']+delta}
 supported_pass=all(r['absolute_pass'] for r in rows) and all(x['simultaneous_upper']<=x['limit'] for x in tests.values())
 supported_fail=any(x['simultaneous_lower']>x['limit'] for x in tests.values())
 bounds[g]={'tests':tests,'point_pass_seeds':sum(r['absolute_pass'] for r in rows),'evidence':'supported_pass_under_t_assumptions' if supported_pass else ('supported_fail_under_t_assumptions' if supported_fail else 'boundary_or_insufficient_evidence')}
terms={name:stats([results[f'{name}_{s}']['terminations']['count'] for s in range(30101,30111)]) for name in ['start','selected']};terms['paired_difference']=stats([results[f'selected_{s}']['terminations']['count']-results[f'start_{s}']['terminations']['count'] for s in range(30101,30111)])
lo,hi=terms['paired_difference']['ci95'];terms['interpretation']='increase_supported_under_t_assumptions' if lo>0 else ('decrease_supported_under_t_assumptions' if hi<0 else 'inconclusive_difference_not_equivalence')
transient_summary={}
for name in ['start','selected']:
 transient_summary[name]={}
 for en in ['release','reverse','load_release','load_reverse','load_full3d']:
  rows=[results[f'{name}_{s}']['transients'][en] for s in range(30101,30111)];rr={}
  for k in ['recovered','censored','recovered_without_reset','any_reset_envs']:
   rr[k]=stats([r['summary'][k] for r in rows])
  for k in ['peak_vector_100ms_N','peak_mag_100ms_N','peak_vector_1s_N','peak_mag_1s_N','peak_vector_100ms_reference_pct','integral_vector_error_2s_Ns','exceed_vector_1p5N_total_s','exceed_vector_1p5N_longest_s']:
   rr[k]={kk:stats([r['summary'][k][kk] for r in rows]) for kk in ['mean','p95','max']}
  rr['settling_start_median_s_all_events']=[r['summary']['settling_start_median_s_all_events'] for r in rows];rr['settling_start_p95_s_all_events']=[r['summary']['settling_start_p95_s_all_events'] for r in rows]
  rr['budget_grid']={b:{dl:stats([r['recovery_start_deadline_budget_grid_pct'][b][dl] for r in rows]) for dl in ['0.1','0.2','0.3','0.5','1','2']} for b in ['0.5','1','1.5','2','3']}
  transient_summary[name][en]=rr
# Fixed checkpoint ONNX comparison on actual newly collected histories, CPU vsCPU.
import onnxruntime as ort
h=np.load(R/'selected_30101.npz')['history_samples'][::10].reshape(-1,25,82);m=models['selected'];torch.onnx.export(m,torch.from_numpy(h[:1]),str(O/'selected.onnx'),input_names=['history'],output_names=['force'],dynamic_axes={'history':{0:'batch'},'force':{0:'batch'}},opset_version=17,dynamo=False)
with torch.no_grad():p=m(torch.from_numpy(h)).numpy()
pred=ort.InferenceSession(str(O/'selected.onnx'),providers=['CPUExecutionProvider']).run(None,{'history':h})[0];error=float(abs(p-pred).max());assert error<1e-5
parity={'max_normalized':error,'max_N':error*80,'pass':True}
rel_sensitivity={}
for g in ['overall','force/2_5','force/5_15','force/low','force/high','force/15_30','force/30_45']:
 rel_sensitivity[g]={}
 for k in ['relative_mag_rms_pct','relative_vector_rms_pct','vector_nrmse_pct']:
  rel_sensitivity[g][k]={str(b):{'point_pass_seeds':sum(results[f'selected_{s}']['groups'][g][k]<=b for s in range(30101,30111)),'budget_status':'hypothetical_not_approved'} for b in [5,10,20]}
decision={'application_error_budget':'undetermined_user_confirmed','application_recovery_budget':'undetermined_user_confirmed','formal_application_acceptance':'undetermined','default_deployment_changed':False,'numerical_pass':True,'absolute_point_all_primary_all_seeds_pass':all(v['point_pass_seeds']==10 for v in bounds.values()),'absolute_simultaneous_all_primary_supported_pass':all(v['evidence'].startswith('supported_pass') for v in bounds.values()),'termination_difference_interpretation':terms['interpretation'],'relative_or_transient_budget_promoted_to_requirement':False,'model':'epoch6320, training stopped6410'}
report={'timing_contract_version':2,'lock':lock,'summary':summary,'paired_differences':paired,'primary_simultaneous_bounds':bounds,'termination_statistics':terms,'transient_summary':transient_summary,'relative_budget_sensitivity':rel_sensitivity,'onnx_parity':parity,'decision':decision,'statistical_target':'equal-seed mean of per-seed metrics; mean of seed P95 is not pooled-frame P95','family_size':K,'confidence_method':'approximate Student-t across10 seeds; Bonferroni two-sided family95% for100 inherited primary tests; not full metrological uncertainty'}
write_json(O/'summary.json',report)
wr=SummaryWriter(str(O/'aggregate'))
for name in ['selected','start']:
 for g,ss in summary[name].items():
  for k,v in ss.items():
   if k in ['n','angle_n','relative_n','zero_n']:continue
   for kk,x in [('seed_mean',v['mean']),('ci95_lower',v['ci95'][0]),('ci95_upper',v['ci95'][1])]:wr.add_scalar(name+'/'+g+'/'+k+'/'+kk,x,10)
wr.add_text('decision',json.dumps(decision,indent=2),10);wr.close();link=TB/'acceptance_v2_aggregate'
if not link.exists():link.symlink_to((O/'aggregate').resolve(),target_is_directory=True)
(O/'COMPLETE').write_text('10 paired seeds,20 runs analyzed; formal budgets unresolved. See summary.json.\n');print('COMPLETE',decision,flush=True)
