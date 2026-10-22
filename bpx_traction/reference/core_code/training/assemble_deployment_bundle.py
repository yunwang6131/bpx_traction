#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--output",required=True); ap.add_argument("--estimator-dir",default="estimators/missing_state_world_v1"); ap.add_argument("--release-status",default="NOT_RELEASED_G4_FAILED_R0_R5_NOT_EXECUTED"); a=ap.parse_args(); root=Path(a.root).resolve(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    sources={
      "policy.onnx":root/"artifacts/gate41_model17488_frozen_20260718/policy.onnx",
      "estimator.onnx":root/a.estimator_dir/"estimator.onnx",
      "estimator.onnx.data":root/a.estimator_dir/"estimator.onnx.data",
      "normalization.npz":root/a.estimator_dir/"normalization.npz",
      "world_force.yaml":root/"experiments/world_force_adapter_v1/config/world_force.yaml",
      "observation_contract.yaml":root/"experiments/world_force_adapter_v1/contracts/observation_contract.yaml",
      "estimator_io.yaml":root/"experiments/world_force_adapter_v1/contracts/estimator_io.yaml",
      "frame_contract.yaml":root/"experiments/world_force_adapter_v1/contracts/frame_contract.yaml",
      "REAL_TEST_PROTOCOL.md":root/"experiments/world_force_adapter_v1/docs/REAL_TEST_PROTOCOL.md",
    }
    copied={}
    for name,src in sources.items():
        if src.exists(): shutil.copy2(src,out/name); copied[name]=src
    manifest={"schema_version":"gate41-deployment-bundle-v4","release_status":a.release_status,"files":{name:sha(out/name) for name in copied},"frozen_policy_sha256":"da33b549c6257beb0323cd09cdfb9c48105b778d9865c93b9878a554346df2a9"}
    (out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__": main()
