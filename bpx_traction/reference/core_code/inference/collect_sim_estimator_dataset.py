#!/usr/bin/env python3
"""Collect deployable inputs and privileged labels from actual MJLab/MuJoCo."""
from __future__ import annotations
import argparse, json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def sensor_vector(raw, actor):
    # Only Go2-deployable signals: IMU, joints, motor torque, last action, command.
    torque=raw.scene["robot"].data.actuator_force
    q_wb=raw.scene["robot"].data.root_link_quat_w
    # The runner's actor observation is 73-D; its first 70 entries are the
    # deployable proprioception.  Zero command entries exactly as deployment
    # does so the estimator cannot see its own generated command.
    proprio = actor[:, :70].clone()
    proprio[:, 13:16] = 0.0
    torque = torque.reshape(torque.shape[0], 12)[:, [3,4,5,0,1,2,9,10,11,6,7,8]]
    return torch.cat((proprio, torque), dim=1)


def targets(raw, actor, tether):
    robot=raw.scene["robot"].data
    signed_log_force=actor[:,58:70].reshape(-1,4,3)
    load=torch.linalg.norm(torch.expm1(torch.abs(signed_log_force)),dim=2)/80.0
    # Deployment policy consumes body-frame force; world-frame force is only
    # used to apply the MuJoCo wrench and must not be used as the label.
    return torch.cat((robot.root_link_lin_vel_w,robot.root_link_pos_w[:,2:3],actor[:,54:58],load,tether.last_force_b),dim=1)


def collect(root, *, seed, num_envs, steps, stride, device, ood=False, jump=False, target=None):
    task="Unitree-Go2-SelfPull-Gate41-World"; cfg=load_env_cfg(task,play=True); cfg.scene.num_envs=num_envs; cfg.seed=seed
    tether_cfg=cfg.events["tether_pull"].params
    tether_cfg["duration_s"]=(0.5,1.5); tether_cfg["cooldown_s"]=(0.2,1.0)
    tether_cfg.update({"static_ramp_release": not jump, "jump_training": jump,
                       "static_cycle_s": 4.0, "static_target_force_range": (10.0, 40.0),
                       "jump_cycle_s": 4.0, "jump_pre_s": 1.0, "jump_post_s": 1.5,
                       "jump_force_range": (12.0, 40.0), "force_frame": "world_fixed",
                       "asset_cfg": tether_cfg["asset_cfg"]})
    if target == "east_hold":
        tether_cfg.update({"static_fixed_force": (20.0, 0.0, 0.0), "static_target_force_range": (20.0, 20.0)})
    elif target == "dir_mag_jump":
        tether_cfg.update({"static_ramp_release": False, "jump_training": True,
                           "jump_fixed_before": (6.0, -10.39230485, 0.0),
                           "jump_fixed_after": (-17.5, 30.31088913, 0.0)})
    if ood:
        tether_cfg["horizontal_force_range"]=(40.0,55.0); tether_cfg["vertical_force_range"]=(-28.0,28.0)
    agent_cfg=load_rl_cfg(task); raw=ManagerBasedRlEnv(cfg=cfg,device=device,render_mode=None); env=RslRlVecEnvWrapper(raw,clip_actions=agent_cfg.clip_actions)
    runner=load_runner_cls(task)(env,asdict(agent_cfg),device=device); runner.load(str(root/"artifacts/gate41_model17488_frozen_20260718/model_17488.pt"),load_cfg={"actor":True},strict=True,map_location=device)
    runner.set_inference_command_source("true_force_teacher"); policy=runner.get_inference_policy(device=device); tether=raw.event_manager.get_term_cfg("tether_pull").func
    obs,_=env.reset(); history=torch.zeros(num_envs,25,82,device=device); xs=[]; ys=[]; env_ids=[]
    for step in range(steps):
        actor=obs["actor"]; history=torch.roll(history,1,1); history[:,0]=sensor_vector(raw,actor)
        runner.set_inference_teacher_force(tether.last_force_b)
        with torch.no_grad(): action=policy(obs); obs,reward,done,info=env.step(action)
        if step>=24 and step%stride==0:
            xs.append(history.detach().cpu().numpy().astype(np.float32)); ys.append(targets(raw,obs["actor"],tether).detach().cpu().numpy().astype(np.float32)); env_ids.append(np.arange(num_envs,dtype=np.int32))
        if done.any(): history[done]=0
    env.close(); return np.concatenate(xs),np.concatenate(ys),np.concatenate(env_ids)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--output",required=True); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--num-envs",type=int,default=256); ap.add_argument("--steps",type=int,default=650); a=ap.parse_args()
    root=Path(a.root).resolve(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); configure_torch_backends(); import mjlab.tasks,src.tasks  # noqa:F401
    x,y,e=collect(root,seed=41801,num_envs=a.num_envs,steps=a.steps,stride=10,device=a.device)
    cut1=int(a.num_envs*.70); cut2=int(a.num_envs*.85)
    masks={"train":e<cut1,"validation":(e>=cut1)&(e<cut2),"test":e>=cut2}
    sizes={}
    for name,mask in masks.items(): np.savez_compressed(out/f"{name}.npz",sensor_history=x[mask],privileged_targets=y[mask]); sizes[name]=int(mask.sum())
    ox,oy,oe=collect(root,seed=41999,num_envs=max(64,a.num_envs//2),steps=max(400,a.steps//2),stride=10,device=a.device,ood=True)
    np.savez_compressed(out/"ood.npz",sensor_history=ox,privileged_targets=oy); sizes["ood"]=len(ox)
    # A separate abrupt-change cohort is kept intact for targeted fine-tuning.
    jx,jy,je=collect(root,seed=42317,num_envs=max(96,a.num_envs),steps=max(600,a.steps),stride=5,device=a.device,jump=True)
    np.savez_compressed(out/"jump.npz",sensor_history=jx,privileged_targets=jy); sizes["jump"]=len(jx)
    manifest={"schema_version":"gate41-estimator-dataset-v3","provenance":"actual_MJLab_MuJoCo","oracle_claim":True,"force_frame":"world_fixed","input_privileged_leakage":False,"input_shape":[25,82],"target_dimension":15,"split_policy":"environment/scenario index; adjacent windows remain in one split","seeds":{"in_distribution":41801,"ood":41999},"sizes":sizes,"notes":"70-D deployable proprioception plus 12 actuator torque estimates. Includes standing zero-force, slow ramp, hold, and release cycles."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main()
