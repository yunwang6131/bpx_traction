#!/usr/bin/env python3
"""真实外力 vs 估计外力的矢量箭头对比图（机身系 xy 平面）。

    conda activate bpx_rma
    python scripts/plot_force_arrows.py                      # 默认：集成模型，各力档抽样
    python scripts/plot_force_arrows.py --model <ts路径> --n 48 --out 图.png

每个小格是一个样本：灰箭头=真实力，红箭头=估计力。两者长度差=幅值误差，夹角=方向误差。
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from bpx_traction import spec  # noqa: E402
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(PROJECT / "logs/traction/v2/ensemble_hinge/estimator.ts"))
    ap.add_argument("--dataset", default=str(PROJECT / "logs/traction/dataset_v2"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(PROJECT / "logs/traction/v2/force_arrows.png"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(Path(args.dataset) / "test.npz")
    H, T = d["history"], d["targets"]
    fb, fe = spec.output_slice("force_b")
    true = T[:, fb:fe]; tmag = np.linalg.norm(true[:, :2], axis=1)

    m = torch.jit.load(args.model, map_location=device).eval()
    preds = []
    with torch.no_grad():
        for b in range(0, len(H), 8192):
            v, _ = m(torch.from_numpy(H[b:b+8192]).to(device)); preds.append(v[:, fb:fe].cpu().numpy())
    pred = np.concatenate(preds)

    bands = [(2,5),(5,10),(10,15),(15,20),(20,30)]
    per = max(1, args.n // len(bands)); rng = np.random.default_rng(args.seed); idx = []
    for lo, hi in bands:
        pool = np.nonzero((tmag>=lo)&(tmag<hi))[0]
        if len(pool): idx.extend(rng.choice(pool, min(per, len(pool)), replace=False))
    idx = np.array(idx)

    cols = 8; rows = int(np.ceil(len(idx)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.0, rows*2.0)); axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        tf, pf = true[i,:2], pred[i,:2]; tm, pm = np.linalg.norm(tf), np.linalg.norm(pf)
        ang = np.degrees(np.arccos(np.clip(tf@pf/(tm*pm+1e-9), -1, 1))); lim = max(tm,pm)*1.25+1
        ax.arrow(0,0,tf[0],tf[1], head_width=lim*0.08, color="0.5", length_includes_head=True, width=lim*0.015)
        ax.arrow(0,0,pf[0],pf[1], head_width=lim*0.08, color="crimson", length_includes_head=True, width=lim*0.015, alpha=0.8)
        ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_aspect("equal")
        ax.axhline(0,color="0.9",lw=0.5); ax.axvline(0,color="0.9",lw=0.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"T{tm:.0f} E{pm:.0f}N d{ang:.0f}", fontsize=8)
    for ax in axes[len(idx):]: ax.axis("off")
    fig.suptitle(f"true(grey) vs est(red) force, body xy  -  {Path(args.model).parent.name}", fontsize=11)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0],[0],color="0.5",lw=3,label="true force"),
                        Line2D([0],[0],color="crimson",lw=3,label="estimated force")],
               loc="lower center", ncol=2, fontsize=10)
    fig.tight_layout(rect=[0,0.03,1,0.97]); fig.savefig(args.out, dpi=110)
    print(f"[traction] {len(idx)} 样本 -> {args.out}")


if __name__ == "__main__":
    main()
