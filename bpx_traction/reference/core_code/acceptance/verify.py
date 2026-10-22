from pathlib import Path
import json,hashlib,numpy as np
from scipy.stats import t
O=Path('logs/rsl_rl/go2_velocity/acceptance_v2');s=json.load(open(O/'summary.json'));rows={p.stem:json.load(open(p)) for p in (O/'runs').glob('*.json')}
assert s['timing_contract_version']==2
assert len(rows)==20 and len(s['lock']['plan'])==10
assert sum(p['order'][0]=='selected' for p in s['lock']['plan'])==5
assert len(set(p['seed'] for p in s['lock']['plan']))==10
assert s['family_size']==100
assert s['decision']['formal_application_acceptance']=='undetermined'
assert not s['decision']['relative_or_transient_budget_promoted_to_requirement'] and not s['decision']['default_deployment_changed']
for seed in range(30101,30111):
 a,b=rows[f'selected_{seed}'],rows[f'start_{seed}'];assert a['gpu']==b['gpu'];assert a['force_plan_sha256']==b['force_plan_sha256']
 for r in [a,b]:
  assert r['audits']['runtime']['physics_audit']['substeps']==24000 and r['audits']['runtime']['physics_audit']['max_wrench_error_N']<1e-6
  assert r['groups']['overall']['n']==644000
  assert r['timing_contract_version']==2
  for ev,rr in r['transients'].items():
   v=rr['summary'];assert v['eligible_envs']==98 and v['recovered']+v['censored']==98
   times=rr['per_environment']['settling_start_s'];assert sum(x is None for x in times)==v['censored']
   for old,physical,confirmed in zip(times,rr['per_environment']['physical_recovery_start_s'],rr['per_environment']['physical_confirmation_s']):
    if old is None:assert physical is None and confirmed is None
    else:assert abs(physical-old-.02)<1e-10 and abs(confirmed-old-.20)<1e-10
   assert v['recovered_without_reset']<=v['recovered']
for g,rs in s['summary']['selected'].items():
 if 'mag_rmse' not in rs:continue
 z=np.array([rows[f'selected_{seed}']['groups'][g]['mag_rmse'] for seed in range(30101,30111)]);st=rs['mag_rmse'];assert abs(z.mean()-st['mean'])<1e-12
 ci=z.mean()+np.array([-1,1])*t.ppf(.975,9)*z.std(ddof=1)/np.sqrt(10);assert np.allclose(ci,st['ci95'])
for g,b in s['primary_simultaneous_bounds'].items():
 for k,v in b['tests'].items():
  sd=s['summary']['selected'][g][k]['sd'];delta=t.ppf(1-.05/200,9)*sd/np.sqrt(10);assert abs(v['simultaneous_upper']-(v['seed_mean']+delta))<1e-12
assert s['onnx_parity']['pass'];assert (O/'COMPLETE').exists()
r={'verified':True,'fresh_paired_seeds':10,'full_runs':20,'physics_substeps':480000,'no_budget_fabrication':True,'recovery_censoring_preserved':True,'t_intervals_recomputed':True,'simultaneous_bounds_recomputed':True,'model_sha256':s['lock']['model_sha256'],'formal_application_acceptance':'undetermined'}
json.dump(r,open(O/'verification.json','w'),indent=2);print(r)
