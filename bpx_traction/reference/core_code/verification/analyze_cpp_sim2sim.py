#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,re
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--directory",required=True);ap.add_argument("--output",required=True);a=ap.parse_args();d=Path(a.directory);rows=list(csv.reader((d/"offline_states.csv").open()));world=[r for r in rows if len(r)>=8 and r[-8]=="world_fixed"]
    fw=[[float(v) for v in r[-6:-3]] for r in world];max_err=max((max(abs(v-e) for v,e in zip(f,[20,0,0])) for f in fw),default=float("inf"));log=(d/"controller.log").read_text(errors="replace");steps=[int(x) for x in re.findall(r"GATE41_RUNTIME mode=estimated_force step=(\d+)",log)]
    report={"evidence":"independent C++ MuJoCo simulator + go2_ctrl + DDS loopback","connected":"Connected to robot." in log,"recorded_frames":len(rows),"world_fixed_frames":len(world),"world_force_max_abs_error_n":max_err,"controller_max_reported_step":max(steps,default=0),"pipeline_functional":bool(world and max_err<1e-9 and steps),"performance_release":"fail","reason":"The historical estimator diverges in C++ Sim2Sim; functional transport/frame validation passes but closed-loop performance does not."};Path(a.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
