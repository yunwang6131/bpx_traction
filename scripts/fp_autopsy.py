#!/usr/bin/env python3
"""零力误报尸检：误报按"力变化后多久"和剖面类型分层。

用 dataset_v2 新增的 since_change / profile_kind 列回答:
误报是集中在松绳后的衰减尾巴上(瞬态假说),还是均匀铺在零力段(表征问题)?
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/home/wangyun/intern/ZGYD/bpx/.claude/worktrees/traction")
sys.path.insert(0, str(PROJECT))
from bpx_traction import spec  # noqa: E402

MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "logs/traction/v2/ensemble/estimator.ts"
DS = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT / "logs/traction/dataset_v2"

d = np.load(DS / "test.npz")
targets = d["targets"]
since = d["since_change"].astype(np.int64)
kind = d["profile_kind"].astype(np.int64)
KINDS = ("hold", "ramp", "release", "reverse", "sweep")

fb, fe = spec.output_slice("force_b")
true_mag = np.linalg.norm(targets[:, fb:fe], axis=1)
zero = true_mag < spec.FORCE_ACTIVE_N
print(f"测试集 {len(targets):,}，零力 {zero.sum():,}（{zero.mean():.0%}）")

model = torch.jit.load(str(MODEL), map_location="cuda").eval()
hist = d["history"][zero]
preds = []
with torch.no_grad():
    for b in range(0, len(hist), 4096):
        v, _ = model(torch.from_numpy(hist[b:b + 4096]).cuda())
        preds.append(v[:, fb:fe].cpu().numpy())
pred = np.concatenate(preds)
fp = np.linalg.norm(pred[:, :2], axis=1)
sz, kz = since[zero], kind[zero]

def row(name, m):
    if m.sum() < 30:
        return
    v = fp[m]
    print(f"  {name:<18} n={m.sum():>7,}  p50 {np.median(v):5.2f}  "
          f"p95 {np.percentile(v, 95):5.2f}  max {v.max():6.2f}  >2N {100*(v>2).mean():5.1f}%")

print("\n== 按距力变化的时间（50 Hz 步）==")
for lo, hi, name in [(0, 25, "0–0.5 s"), (25, 50, "0.5–1 s"), (50, 100, "1–2 s"),
                     (100, 200, "2–4 s"), (200, 10**9, ">4 s")]:
    row(name, (sz >= lo) & (sz < hi))

print("\n== 按剖面类型 ==")
for k, name in enumerate(KINDS):
    row(name, kz == k)

print("\n== 交叉：release 剖面 × 时间 ==")
rel = kz == KINDS.index("release")
for lo, hi, name in [(0, 25, "release 0–0.5 s"), (25, 50, "release 0.5–1 s"),
                     (50, 100, "release 1–2 s"), (100, 10**9, "release >2 s")]:
    row(name, rel & (sz >= lo) & (sz < hi))

print("\n== hold 剖面的纯稳态零力（变化后 >4 s，相当于'机器人走路没人碰'）==")
row("hold >4 s", (kz == KINDS.index("hold")) & (sz > 200))
