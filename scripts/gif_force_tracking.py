#!/usr/bin/env python3
"""动态回放（诊断版）：真实力 vs 估计力，记录姿态/摔倒/世界力做核对。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from isaaclab.app import AppLauncher  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="logs/rsl_rl/bpx_pro_sprint4/2026-10-21_11-02-37_student")
ap.add_argument("--estimator", default=str(PROJECT/"logs/traction/v2/ensemble_hinge/estimator.ts"))
ap.add_argument("--task", default="Isaac-Velocity-Flat-BpxPro-v0")
ap.add_argument("--agent", default="rsl_rl_distillation_cfg_entry_point")
ap.add_argument("--walk", type=float, default=0.5, help="前进指令 m/s（训练时多数带速度）")
ap.add_argument("--peak", type=float, default=15.0, help="峰值力 N（站立别用太大会推倒）")
ap.add_argument("--out", default=str(PROJECT/"logs/traction/v2/force_tracking.npz"))
AppLauncher.add_app_launcher_args(ap); ap.set_defaults(headless=True)
args = ap.parse_args()
app = AppLauncher(args).app
import gymnasium as gym, numpy as np, torch  # noqa: E402
from importlib.metadata import version as _ver  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402
import bpx_rma  # noqa: E402,F401
from bpx_traction import spec  # noqa: E402
from bpx_traction.env.frame import FrameAssembler  # noqa: E402
from bpx_traction.env.tether import TetherApplier  # noqa: E402
from bpx_traction.env import traction_cfg  # noqa: E402
from bpx_traction.data import scenarios  # noqa: E402
def acfg(): return handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task,args.agent), _ver("rsl-rl-lib"))
def ckpt(raw):
    p=Path(raw)
    if not p.is_absolute(): p=(PROJECT/p).resolve()
    return p if p.is_file() else Path(get_checkpoint_path(str(p.parent),p.name,"model_.*.pt"))
NENV=16
ec=load_cfg_from_registry(args.task,"env_cfg_entry_point")
traction_cfg.prepare_for_traction(ec,num_envs=NENV,episode_length_s=60.0)
env=RslRlVecEnvWrapper(gym.make(args.task,cfg=ec),clip_actions=acfg().clip_actions)
base=env.unwrapped; dt=base.step_dt; robot=base.scene["robot"]
frames=FrameAssembler(base); tether=TetherApplier(robot,warn_unconfirmed=False); tether.reset_offsets()
est=torch.jit.load(args.estimator,map_location=base.device).eval()
hist=torch.zeros(NENV,spec.HISTORY_LENGTH,spec.FRAME_SIZE,device=base.device)
# 剖面：settle→ramp→hold→release→reverse→hold→release。world系方位角30°
prof=scenarios.TetherProfile("reverse",scenarios.ForceSpec(args.peak,30.0,0.0),settle_s=1.0,ramp_s=1.2,hold_s=2.0,tail_s=4.0)
steps=prof.num_steps()
ag=acfg(); isd=ag.class_name=="DistillationRunner"
runner=(DistillationRunner if isd else OnPolicyRunner)(env,ag.to_dict(),log_dir=None,device=ag.device)
cp=ckpt(args.policy)
runner.load(str(cp),load_cfg={"student":True,"teacher":True,"iteration":False}) if isd else runner.load(str(cp))
policy=runner.get_inference_policy(device=ag.device)
cmd=torch.zeros(NENV,3,device=base.device); cmd[:,0]=args.walk
tether.clear(); base.reset(); obs=env.get_observations()
f0=frames.frame(); hist[:]=f0.unsqueeze(1).expand(-1,spec.HISTORY_LENGTH,-1)
fb,fe=spec.output_slice("force_b")
true_log,est_log,worldf_log,rp_log,term_log,valid=[],[],[],[],[],[]
vstep=torch.zeros(NENV,dtype=torch.long,device=base.device)
prev_fw=torch.zeros(NENV,3,device=base.device)
with torch.no_grad():
    for k in range(steps):
        base.command_manager.get_command("base_velocity")[:]=cmd
        fw=torch.tensor(prof.force_at(k*dt),dtype=torch.float32,device=base.device).repeat(NENV,1)
        tether.apply_world(fw)
        # 标签同训练：上一步力转当前机身系
        label_b=quat_apply_inverse(robot.data.root_quat_w, prev_fw)
        fr=frames.frame(); hist=torch.roll(hist,-1,1); hist[:,-1]=fr
        vstep=torch.clamp(vstep+1,max=spec.HISTORY_LENGTH)
        v,_=est(hist); ef=v[:,fb:fe]
        # roll/pitch
        g=robot.data.projected_gravity_b
        roll=torch.atan2(g[:,1],-g[:,2]); pitch=torch.atan2(-g[:,0],-g[:,2])
        true_log.append(label_b[0,:2].cpu().numpy()); est_log.append(ef[0,:2].cpu().numpy())
        worldf_log.append(fw[0,:2].cpu().numpy())
        rp_log.append([float(roll[0]),float(pitch[0])]); 
        valid.append(int(vstep[0]>=spec.HISTORY_LENGTH))
        prev_fw=fw
        a=policy(obs); obs,_,dn,_=env.step(a); term_log.append(int(dn[0]))
        if bool(dn.any()):
            idx=torch.nonzero(dn.bool()).flatten(); hist[idx]=frames.frame()[idx].unsqueeze(1).expand(-1,spec.HISTORY_LENGTH,-1); vstep[idx]=0
tether.clear(); env.close()
np.savez(args.out,true=np.array(true_log),est=np.array(est_log),worldf=np.array(worldf_log),
         rollpitch=np.array(rp_log),term=np.array(term_log),valid=np.array(valid),dt=dt)
print(f"[traction] {len(true_log)}帧 -> {args.out}")
app.close()
