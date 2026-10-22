#!/usr/bin/env python3
"""Run group C: deployable estimator -> adapter -> frozen Actor in MJLab."""
from __future__ import annotations
import argparse,json
from dataclasses import asdict
from pathlib import Path
import numpy as np,torch,yaml
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");ap.add_argument("--output",required=True);ap.add_argument("--model-dir",default="estimators/missing_state_world_v2");ap.add_argument("--min-confidence",type=float,default=.55);ap.add_argument("--min-velocity-confidence",type=float,default=0.0);ap.add_argument("--seed",type=int,default=41001);ap.add_argument("--device",default="cuda:0");ap.add_argument("--scenario-id",default=None);ap.add_argument("--sensor-noise",type=float,default=0.0,help="relative sensor noise scale");ap.add_argument("--sensor-bias",type=float,default=0.0,help="constant relative bias scale");ap.add_argument("--random-walk",type=float,default=0.0,help="per-step relative random-walk scale");ap.add_argument("--yaw-drift-deg",type=float,default=0.0,help="total injected yaw drift over one scenario");ap.add_argument("--dropout-prob",type=float,default=0.0);ap.add_argument("--delay-steps",type=int,default=0);ap.add_argument("--oracle-force",action="store_true");ap.add_argument("--oracle-state",action="store_true");a=ap.parse_args();root=Path(a.root).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    configure_torch_backends();import mjlab.tasks,src.tasks  # noqa:F401
    # Exercise the complete fixed scenario list, including yaw-90/diagonal
    # world-force cases and release transitions.  The world-fixed force is
    # applied independently of the robot's randomized initial heading.
    scenarios=yaml.safe_load((root/"experiments/world_force_adapter_v1/config/scenarios.yaml").read_text())["scenarios"]
    if a.scenario_id is not None: scenarios=[scenario for scenario in scenarios if scenario["id"]==a.scenario_id]
    task="Unitree-Go2-SelfPull-Gate41-World";cfg=load_env_cfg(task,play=True);cfg.scene.num_envs=1;cfg.seed=a.seed;agent_cfg=load_rl_cfg(task)
    raw=ManagerBasedRlEnv(cfg=cfg,device=a.device,render_mode=None);env=RslRlVecEnvWrapper(raw,clip_actions=agent_cfg.clip_actions);runner=load_runner_cls(task)(env,asdict(agent_cfg),device=a.device);runner.load(str(root/"artifacts/gate41_model17488_frozen_20260718/model_17488.pt"),load_cfg={"actor":True},strict=True,map_location=a.device)
    actor_policy=runner.alg.get_policy().to(a.device);estimator=torch.jit.load(str(root/a.model_dir/"estimator.ts"),map_location=a.device).eval();tether=raw.event_manager.get_term_cfg("tether_pull").func
    records=[]; sensor_records=[]; value_records=[]; actor_records=[]; target_records=[]; diagnostic_records=[]
    frame_map=[];sample_index=0
    for scenario_index,scenario in enumerate(scenarios):
        frame_id=f"W0-{a.seed}-{scenario['id']}";frame_map.append({"scenario_index":scenario_index,"scenario_id":scenario["id"],"frame_id":frame_id})
        obs,_=env.reset();history=torch.zeros(1,25,64,device=a.device);requested=torch.tensor(scenario["force_w_n"],device=a.device,dtype=torch.float32);steps=round(float(scenario["duration_s"])/raw.step_dt);sensor_queue=[];prev_force_b=torch.zeros(1,3,device=a.device);sensor_bias=None;random_walk=torch.zeros(1,64,device=a.device)
        # The deployed estimator expects a full temporal window.  Seed it with
        # the first measured sensor frame instead of an out-of-distribution
        # all-zero history after every reset.
        initial_actor=obs["actor"]
        initial_torque=raw.scene["robot"].data.actuator_force
        initial_q=raw.scene["robot"].data.root_link_quat_w
        initial_sensor=torch.cat((initial_actor[:,0:3],initial_actor[:,6:9],initial_actor[:,9:12],initial_q,initial_actor[:,18:30],initial_actor[:,30:42],initial_torque,initial_actor[:,42:54],initial_actor[:,13:16]),1)
        history[:]=initial_sensor[:,None,:]
        for step in range(steps):
            active="release_at_s" not in scenario or step*raw.step_dt<float(scenario["release_at_s"]);force_w=requested if active else torch.zeros_like(requested);tether.set_manual_force(force_w,True)
            raw_actor=obs["actor"];torque=raw.scene["robot"].data.actuator_force;q_input=raw.scene["robot"].data.root_link_quat_w;sensor=torch.cat((raw_actor[:,0:3],raw_actor[:,6:9],raw_actor[:,9:12],q_input,raw_actor[:,18:30],raw_actor[:,30:42],torque,raw_actor[:,42:54],raw_actor[:,13:16]),1)
            sensor_for_est=sensor
            sensor_scale=torch.clamp(sensor.abs(),min=1.0)
            if sensor_bias is None: sensor_bias=a.sensor_bias*torch.randn_like(sensor)*sensor_scale
            if a.sensor_bias>0: sensor_for_est=sensor_for_est+sensor_bias
            if a.random_walk>0:
                random_walk=random_walk+a.random_walk*torch.randn_like(sensor)*sensor_scale
                sensor_for_est=sensor_for_est+random_walk
            if a.delay_steps>0:
                sensor_queue.append(sensor.clone())
                sensor_for_est=sensor_queue.pop(0) if len(sensor_queue)>a.delay_steps else sensor_queue[0]
            if a.sensor_noise>0:
                sensor_for_est=sensor_for_est + a.sensor_noise*torch.randn_like(sensor_for_est)*torch.clamp(sensor_for_est.abs(),min=1.0)
            if a.dropout_prob>0:
                keep=torch.rand_like(sensor_for_est)>a.dropout_prob
                sensor_for_est=torch.where(keep,sensor_for_est,history[:,0])
            if a.yaw_drift_deg!=0:
                angle=torch.tensor(np.deg2rad(a.yaw_drift_deg)*(step/max(steps-1,1)),device=a.device,dtype=sensor_for_est.dtype);c=torch.cos(angle/2);s=torch.sin(angle/2);qw,qx,qy,qz=(sensor_for_est[:,i].clone() for i in range(9,13))
                sensor_for_est[:,9]=c*qw-s*qz;sensor_for_est[:,10]=c*qx-s*qy;sensor_for_est[:,11]=c*qy+s*qx;sensor_for_est[:,12]=c*qz+s*qw
            qn=sensor_for_est[:,9:13];sensor_for_est[:,9:13]=qn/torch.clamp(torch.linalg.norm(qn,dim=1,keepdim=True),min=1e-6)
            history=torch.roll(history,1,1);history[:,0]=sensor_for_est
            with torch.no_grad(): values,confidence=estimator(history)
            sensor_records.append(history[0].detach().cpu().numpy().copy())
            value_records.append(torch.cat((values[0], confidence[0])).detach().cpu().numpy().copy())
            fw_record = force_w if force_w.ndim == 1 else force_w[0]
            robot_now=raw.scene["robot"].data
            signed_log_force=raw_actor[:,58:70].reshape(-1,4,3)
            load_now=torch.linalg.norm(torch.expm1(torch.abs(signed_log_force)),dim=2)/80.0
            target_full=torch.cat((robot_now.root_link_lin_vel_w,robot_now.root_link_pos_w[:,2:3],raw_actor[:,54:58],load_now,force_w.reshape(1,3)),1)
            target_records.append(target_full[0].detach().cpu().numpy().copy())
            q=raw.scene["robot"].data.root_link_quat_w;vel_b=quat_apply_inverse(q,values[:,:3]);
            if a.oracle_state:
                vel_b=raw_actor[:,3:6]; height_for_actor=raw_actor[:,12:13]; contact_for_actor=raw_actor[:,54:58]; loads_for_actor=raw_actor[:,58:70]
            else:
                height_for_actor=values[:,3:4]; contact_for_actor=values[:,4:8]; loads_for_actor=None
            force_source_w=force_w if a.oracle_force else values[:,12:15]
            force_b_raw=quat_apply_inverse(q,force_source_w);trusted=confidence[:,3:4]>=.55;force_b_raw=torch.where((trusted|a.oracle_force),force_b_raw,torch.zeros_like(force_b_raw))
            # Independent force slew limiter only engages on a large
            # one-cycle jump; normal changes remain responsive to the tether.
            delta=torch.linalg.norm(force_b_raw-prev_force_b,dim=1,keepdim=True)
            alpha=torch.where((delta>15.0)&(torch.linalg.norm(prev_force_b,dim=1,keepdim=True)>1.0),torch.full_like(delta,0.35),torch.ones_like(delta))
            force_b=prev_force_b+alpha*(force_b_raw-prev_force_b);prev_force_b=force_b.detach()
            force_for_actor=force_b
            p=raw_actor[:,:-3].clone();p[:,3:6]=vel_b;p[:,12:13]=height_for_actor;p[:,13:15]=runner._follow_command(force_for_actor/80.0);p[:,15]=0;p[:,16:18]=runner._phase_from_command(p[:,13:15]);p[:,54:58]=contact_for_actor
            if loads_for_actor is None:
                loads=torch.zeros(1,4,3,device=a.device);loads[:,:,2]=-torch.log1p(values[:,8:12]*80.0);p[:,58:70]=loads.flatten(1)
            else: p[:,58:70]=loads_for_actor
            actor=torch.cat((p,force_for_actor/80.0),1);actor_obs=obs.clone();actor_obs["actor"]=actor
            actor_records.append(actor[0].detach().cpu().numpy().copy())
            with torch.no_grad(): actions=actor_policy(actor_obs)
            # Independent safety gate: sustained low force confidence removes
            # policy authority and requests the action-manager default stand.
            low_confidence=(confidence[:,3:4] < a.min_confidence)|(confidence[:,0:1] < a.min_velocity_confidence)
            actions=torch.where(low_confidence,torch.zeros_like(actions),actions)
            with torch.no_grad(): obs,reward,done,info=env.step(actions)
            robot=raw.scene["robot"].data;records.append(np.concatenate((np.array([a.seed,step,step*raw.step_dt,float(done.item())],np.float32),force_w.cpu().numpy(),values[0,12:15].cpu().numpy(),confidence[0].cpu().numpy(),robot.root_link_pos_w[0].cpu().numpy(),robot.root_link_lin_vel_w[0].cpu().numpy(),actions[0].cpu().numpy(),force_for_actor[0].detach().cpu().numpy(),np.array([scenario_index,sample_index],np.float32))))
            contact=raw.scene["feet_ground_contact"].data.found[0].float()
            diagnostic_records.append(np.concatenate((
                robot.root_link_quat_w[0].detach().cpu().numpy(),
                robot.root_link_ang_vel_w[0].detach().cpu().numpy(),
                robot.joint_pos[0].detach().cpu().numpy(),
                robot.joint_vel[0].detach().cpu().numpy(),
                robot.actuator_force[0].detach().cpu().numpy(),
                robot.site_pos_w[0].detach().cpu().numpy().reshape(-1),
                robot.site_lin_vel_w[0].detach().cpu().numpy().reshape(-1),
                contact.detach().cpu().numpy().reshape(-1),
            )))
            sample_index+=1
            if done.item(): history.zero_()
    env.close();arr=np.stack(records);name=f"estimated_world_seed{a.seed}.npz";np.savez_compressed(out/name,estimator_in_loop_rollout=arr,sensor_history=np.stack(sensor_records),predicted_values_confidence=np.stack(value_records),privileged_targets=np.stack(target_records),target_debug=np.stack(target_records),actor_input=np.stack(actor_records),extended_diagnostics=np.stack(diagnostic_records));manifest={"schema_version":"gate41-estimator-loop-v7","group":"C","physics":"MJLab/MuJoCo","seed":a.seed,"file":name,"samples":len(arr),"terminated_samples":int(arr[:,3].sum()),"sensor_shape":[25,64],"target_shape":[15],"input_privileged_leakage":False,"force_slew_limit":"large jump only: if delta>15 N and previous norm>1 N, apply 35% of delta; otherwise pass through","diagnostic_oracle_force":a.oracle_force,"diagnostic_oracle_state":a.oracle_state,"perturbations":{"sensor_noise":a.sensor_noise,"sensor_bias":a.sensor_bias,"random_walk_per_step":a.random_walk,"yaw_drift_deg_per_scenario":a.yaw_drift_deg,"dropout_probability":a.dropout_prob,"delay_steps":a.delay_steps},"scenario_frames":frame_map,"columns":["seed","step","time_s","terminated","true_force_w_x","true_force_w_y","true_force_w_z","estimated_force_w_x","estimated_force_w_y","estimated_force_w_z","confidence_velocity","confidence_height","confidence_contact","confidence_force","pos_x","pos_y","pos_z","vel_x","vel_y","vel_z",*[f"action_{i}" for i in range(12)],"actor_force_b_x","actor_force_b_y","actor_force_b_z","scenario_index","sample_index"],"extended_diagnostics":{"description":"privileged evaluation-only fields; never estimator or Actor input","columns":["quat_w","quat_x","quat_y","quat_z","ang_vel_w_x","ang_vel_w_y","ang_vel_w_z",*[f"joint_pos_{i}" for i in range(12)],*[f"joint_vel_{i}" for i in range(12)],*[f"actuator_force_{i}" for i in range(12)],*[f"foot_pos_w_{foot}_{axis}" for foot in ("FR","FL","RR","RL") for axis in ("x","y","z")],*[f"foot_vel_w_{foot}_{axis}" for foot in ("FR","FL","RR","RL") for axis in ("x","y","z")],*[f"foot_contact_{foot}" for foot in ("FR","FL","RR","RL")]]}};(out/f"manifest_seed{a.seed}.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
