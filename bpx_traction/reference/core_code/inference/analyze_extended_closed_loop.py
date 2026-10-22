#!/usr/bin/env python3
"""Compute directly observable closed-loop metrics from v7 rollout logs."""
import argparse, json
from pathlib import Path
import numpy as np


def quat_roll_pitch(q):
    w,x,y,z=np.moveaxis(q,-1,0)
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    return np.rad2deg(roll),np.rad2deg(pitch)


def point_segment_distance(p,a,b):
    ab=b-a; den=float(ab@ab)
    if den<1e-12:return float(np.linalg.norm(p-a))
    t=np.clip(float((p-a)@ab)/den,0,1)
    return float(np.linalg.norm(p-(a+t*ab)))


def support_margin(p, feet, contact):
    pts=feet[contact][:,:2]
    if len(pts)==0:
        return np.nan
    if len(pts)<3:
        return -min(point_segment_distance(p,pts[i],pts[min(i+1,len(pts)-1)]) for i in range(len(pts)))
    c=pts.mean(0); order=np.argsort(np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0])); poly=pts[order]
    cross=[];dist=[]
    for i,a in enumerate(poly):
        b=poly[(i+1)%len(poly)]; e=b-a
        cross.append(e[0]*(p[1]-a[1])-e[1]*(p[0]-a[0]));dist.append(point_segment_distance(p,a,b))
    inside=all(v>=-1e-9 for v in cross) or all(v<=1e-9 for v in cross)
    return min(dist)*(1 if inside else -1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('paths',nargs='+');ap.add_argument('--output',required=True);a=ap.parse_args()
    runs=[]
    for path in map(Path,a.paths):
        z=np.load(path); r=z['estimator_in_loop_rollout']; d=z['extended_diagnostics']; actor=z['actor_input']; dt=float(np.median(np.diff(r[:,2])))
        q=d[:,:4]; roll,pitch=quat_roll_pitch(q); torque=d[:,31:43]; feet=d[:,43:55].reshape(-1,4,3); foot_vel=d[:,55:67].reshape(-1,4,3); contact=d[:,67:71]>0
        slip=np.linalg.norm(foot_vel[:,:,:2],axis=2); contact_count=max(int(contact.sum()),1)
        margins=np.array([support_margin(r[i,14:16],feet[i],contact[i]) for i in range(len(r))])
        cmd=actor[:,13:15]; vel=r[:,17:19]; action=r[:,20:32]
        action_rate=np.diff(action,axis=0)/dt
        defined=np.isfinite(margins)
        runs.append({'file':path.name,'samples':len(r),'terminated_samples':int(r[:,3].sum()),'roll_abs_peak_deg':float(np.max(np.abs(roll))),'pitch_abs_peak_deg':float(np.max(np.abs(pitch))),'joint_torque_abs_peak_nm':float(np.max(np.abs(torque))),'velocity_tracking_rmse_mps':float(np.sqrt(np.mean((vel-cmd)**2))),'action_rate_abs_p95_per_s':float(np.percentile(np.abs(action_rate),95)),'foot_slip_contact_rate_gt_0p1_mps':float(((slip>0.1)&contact).sum()/contact_count),'foot_slip_velocity_mean_mps':float(slip[contact].mean()) if contact.any() else None,'support_margin_defined_rate':float(defined.mean()),'support_margin_min_m':float(np.nanmin(margins)) if defined.any() else None,'support_margin_negative_rate_when_defined':float((margins[defined]<0).mean()) if defined.any() else None})
    out={'schema_version':'gate41-extended-closed-loop-v1','metric_definitions':{'foot_slip_contact_rate_gt_0p1_mps':'fraction of contacting foot-samples with horizontal foot speed > 0.1 m/s','support_margin':'signed planar distance from base projection to contact-foot convex hull; negative outside or fewer than three contacts'},'runs':runs}
    Path(a.output).write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
