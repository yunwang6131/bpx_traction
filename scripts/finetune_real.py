#!/usr/bin/env python3
"""用真机无外力录制微调力估计器，压误报。

    conda activate bpx_rma
    python scripts/finetune_real.py --init logs/traction/estimator_dilated \
                                    --real logs/traction/real_finetune/real.npz

## 要解决的问题

仿真里训出来的估计器在真机上误报很大：同一个指标（零力样本的 ‖F̂_xy‖ p95）
仿真 4.87 N → 真机 32.22 N，**放大 6.6 倍**。设计力范围上限只有 30 N，误报已经
盖过整个可用区间，定不出死区。

诊断（`scripts/real_deadzone.py` + 相关性分析）：

* 误报**不是**固定偏置（|均值|/std = 0.22），也**不是**步态抖动（步态频带只占 7.9%
  能量），而是 0.3–0.9 Hz 的慢漂移 —— 所以 0.2 s 去抖穿得过去。
* 相关性最强的是**"估计速度 − 指令速度"，r = +0.759**；速度头在真机上低估 61%
  （走路时估 0.264 vs 指令 0.683 m/s）。
* 力头和速度头共用主干。主干的内部动力学模型在真机上不准，网络把这个差额归因成外力——
  在仿真里这个归因是对的（牵引绳是唯一的外部扰动源），真机上就全错了。

## 怎么训

每一步同时走两个批次：

* **仿真批次**：原来那套全监督（`losses.estimator_loss`），力桶配平。它守住"有力的时候
  要能报出来"，防止网络退化成"永远输出零"。
* **真机批次**：`losses.real_anchor_loss` —— 力 → 0（确定标签）+ 速度 → 低通指令
  （弱标签）。后者才是对因的那一项。

## 怎么选模型

**不能只看真机误报降没降** —— 一个什么都不输出的网络误报是 0。所以选模型用的是
带约束的准则：

    在"仿真 xy RMSE 相对初始值劣化不超过 --max-sim-degradation"的 epoch 里，
    挑真机零力 p99 最低的那个。

另外每轮都报仿真**高力段**的 RMSE：网络一旦不响应力，这个数会先炸。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from bpx_traction import losses, metrics, spec  # noqa: E402
from bpx_traction.model import (NormalizedForceTCN, count_parameters,  # noqa: E402
                                from_checkpoint)
from bpx_traction.normalize import Normalizer  # noqa: E402


def forward(model, x, mean, std, device, batch=4096):
    out_v, out_c = [], []
    model.eval()
    with torch.no_grad():
        for a in range(0, len(x), batch):
            chunk = torch.from_numpy(x[a:a + batch]).to(device)
            v, c = model((chunk - mean) / std)
            out_v.append(v.cpu().numpy())
            out_c.append(c.cpu().numpy())
    model.train()
    return np.concatenate(out_v), np.concatenate(out_c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, default=spec.DEFAULT_ESTIMATOR_DIR,
                    help="从哪个检查点开始微调")
    ap.add_argument("--real", type=Path,
                    default=spec.TRACTION_LOG_DIR / "real_finetune" / "real.npz")
    ap.add_argument("--dataset", type=Path, default=None,
                    help="仿真数据集，默认跟随 --init 训练时用的那个")
    ap.add_argument("--out", type=Path, default=spec.TRACTION_LOG_DIR / "estimator_real_ft")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256, help="仿真批次大小")
    ap.add_argument("--real-batch", type=int, default=256, help="真机批次大小")
    ap.add_argument("--lr", type=float, default=1e-4, help="微调用小学习率")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--real-weight", type=float, default=1.0, help="真机那一支的总权重 λ")
    ap.add_argument("--anchor-force", type=float, default=1.0)
    ap.add_argument("--anchor-velocity", type=float, default=2.0)
    ap.add_argument("--max-sim-degradation", type=float, default=0.20,
                    help="仿真 xy RMSE 允许相对初始值劣化多少，超了的 epoch 不参与选型")
    ap.add_argument("--holdout", type=float, default=0.0,
                    help="真机数据按时间留出后面这个比例只用来评估，不参与训练。"
                         "0 = 全部用于训练（此时报的真机指标是**样本内**的，会偏乐观）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---------------- 模型 ----------------
    ck = torch.load(args.init / "estimator.pt", map_location="cpu", weights_only=False)
    model = from_checkpoint(ck).to(device)      # 按检查点记的结构，别手搓
    model.train()
    print(f"[ft] 初始模型 {args.init.name}  {count_parameters(model):,} 参数  "
          f"residual={model.residual}  空洞 {model.dilations}  "
          f"仿真 xy RMSE {ck['score_xy_rmse_n']:.3f} N")

    # 归一化必须沿用初始模型那一份 —— 换了的话权重全错位
    normalizer = Normalizer.load(args.init / "normalization.npz")
    mean = torch.from_numpy(normalizer.mean).to(device)
    std = torch.from_numpy(normalizer.std).to(device)

    # ---------------- 仿真数据 ----------------
    root = args.dataset
    if root is None:
        for cand in sorted(spec.TRACTION_LOG_DIR.glob("dataset*")):
            nz = cand / "normalization.npz"
            if nz.is_file() and np.allclose(np.load(nz)["mean"], normalizer.mean):
                root = cand
                break
    if root is None:
        raise SystemExit("找不到和初始模型归一化一致的仿真数据集，用 --dataset 指定")
    print(f"[ft] 仿真数据集 {root.name}（归一化和初始模型一致）")

    tr = np.load(root / "train.npz")
    va = np.load(root / "validation.npz")
    sim_x = torch.from_numpy(tr["history"].astype(np.float32))
    sim_y = torch.from_numpy(tr["targets"].astype(np.float32))
    buckets = losses.bucket_indices(sim_y)

    # ---------------- 真机数据 ----------------
    rd = np.load(args.real)
    n_real = len(rd["history"])
    # 按**时间**切，不要随机切：相邻窗口重叠 24 帧，随机切会让训练集和评估集
    # 共享几乎相同的样本，评出来的数毫无意义。
    cut = int(round(n_real * (1.0 - args.holdout)))
    tr_sl, ev_sl = slice(0, cut), (slice(cut, n_real) if args.holdout > 0 else slice(0, n_real))
    real_x = torch.from_numpy(rd["history"][tr_sl].astype(np.float32))
    real_v = torch.from_numpy(rd["vel_label"][tr_sl].astype(np.float32))
    real_w = torch.from_numpy(rd["vel_weight"][tr_sl].astype(np.float32))
    print(f"[ft] 仿真 {len(sim_x):,} 训练窗；真机 {n_real:,} 窗"
          f"（{n_real/spec.POLICY_HZ:.0f} s，力真值全 0）")
    if args.holdout > 0:
        print(f"[ft] 真机按时间留出后 {args.holdout:.0%}："
              f"训练 {cut:,} 窗 / 评估 {n_real - cut:,} 窗（**样本外**）")
    else:
        print("[ft] ⚠ 真机数据全部用于训练，下面报的真机指标是**样本内**的，偏乐观")

    # ---------------- 评估口径 ----------------
    fb, fe = spec.output_slice("force_b")
    vb, _ve = spec.output_slice("base_lin_vel_b")
    val_x, val_y = va["history"].astype(np.float32), va["targets"].astype(np.float32)
    true_mag = np.linalg.norm(val_y[:, fb:fb + 2], axis=1)
    sim_zero = true_mag < spec.FORCE_ACTIVE_N
    sim_high = true_mag > 20.0
    real_np = rd["history"][ev_sl].astype(np.float32)
    real_cmd = rd["cmd"][ev_sl]
    real_walk = np.abs(real_cmd[:, 0]) > 0.3

    def assess():
        pv, _pc = forward(model, val_x, mean, std, device)
        rep = metrics.force_metrics(pv[:, fb:fe], val_y[:, fb:fe])
        rv, _rc = forward(model, real_np, mean, std, device)
        rmag = np.linalg.norm(rv[:, fb:fb + 2], axis=1)
        ratio = float(rv[real_walk, vb].mean() / real_cmd[real_walk, 0].mean())
        return {
            "sim_xy_rmse": rep["xy_vector_rmse"],
            "sim_zero_p95": float(np.percentile(
                np.linalg.norm(pv[sim_zero, fb:fb + 2], axis=1), 95)),
            "sim_high_rmse": float(np.sqrt((np.linalg.norm(
                pv[sim_high, fb:fb + 2] - val_y[sim_high, fb:fb + 2], axis=1) ** 2).mean())),
            "real_p95": float(np.percentile(rmag, 95)),
            "real_p99": float(np.percentile(rmag, 99)),
            "real_max": float(rmag.max()),
            "real_vel_ratio": ratio,
        }

    base = assess()
    limit = base["sim_xy_rmse"] * (1.0 + args.max_sim_degradation)
    print(f"[ft] 起点：仿真 xy {base['sim_xy_rmse']:.3f}  高力段 {base['sim_high_rmse']:.3f}  "
          f"仿真零力 p95 {base['sim_zero_p95']:.2f}\n"
          f"     真机 p95 {base['real_p95']:.2f}  p99 {base['real_p99']:.2f}  "
          f"max {base['real_max']:.2f} N  速度头/指令 {base['real_vel_ratio']:.2f}")
    print(f"[ft] 选型约束：仿真 xy RMSE ≤ {limit:.3f}（劣化 ≤{args.max_sim_degradation:.0%}），"
          f"在此之内挑真机 p99 最低的\n")

    # ---------------- 训练 ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    w = losses.LossWeights()
    anchor = losses.RealAnchorWeights(force_zero=args.anchor_force,
                                      velocity=args.anchor_velocity)
    gen = torch.Generator().manual_seed(args.seed)

    best = {"real_p99": float("inf"), "epoch": -1}
    log, started = [], time.time()
    for epoch in range(1, args.epochs + 1):
        acc = {}
        for _ in range(args.steps_per_epoch):
            si = losses.sample_balanced_batch(buckets, args.batch_size, gen)
            sx = (sim_x[si].to(device) - mean) / std
            sy = sim_y[si].to(device)
            sv, sc = model(sx)
            sim_loss, detail = losses.estimator_loss(sv, sc, sy, w)

            ri = torch.randint(len(real_x), (args.real_batch,), generator=gen)
            rx = (real_x[ri].to(device) - mean) / std
            rv, _rc = model(rx)
            real_loss, rdetail = losses.real_anchor_loss(
                rv, real_v[ri].to(device), real_w[ri].to(device), w, anchor)

            loss = sim_loss + args.real_weight * real_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in {**detail, **rdetail}.items():
                acc[k] = acc.get(k, 0.0) + v / args.steps_per_epoch

        m = assess()
        row = {"epoch": epoch, **m, **{f"loss_{k}": v for k, v in acc.items()}}
        log.append(row)
        ok = m["sim_xy_rmse"] <= limit
        star = ""
        if ok and m["real_p99"] < best["real_p99"] - 1e-6:
            best = {**m, "epoch": epoch}
            torch.save({"model": model.state_dict(), "channels": model.channels,
                        "blocks": model.num_blocks, "dilations": model.dilations,
                        "input_dim": model.input_dim, "residual": model.residual,
                        "epoch": epoch, "score_xy_rmse_n": m["sim_xy_rmse"],
                        "real_p99_n": m["real_p99"]}, args.out / "estimator.pt")
            star = "  ★"
        print(f"  epoch {epoch:3d}  仿真 xy {m['sim_xy_rmse']:6.3f}{'' if ok else ' ✗超限'}  "
              f"高力 {m['sim_high_rmse']:6.3f}  "
              f"真机 p95 {m['real_p95']:6.2f}  p99 {m['real_p99']:6.2f}  "
              f"速度比 {m['real_vel_ratio']:.2f}{star}")

    if best["epoch"] < 0:
        raise SystemExit("没有任何 epoch 满足仿真劣化约束 —— 放宽 --max-sim-degradation "
                         "或调低 --real-weight")

    # ---------------- 导出 ----------------
    ck_best = torch.load(args.out / "estimator.pt", map_location="cpu", weights_only=False)
    final = from_checkpoint(ck_best)
    mean_t = torch.from_numpy(normalizer.mean).float()
    std_t = torch.from_numpy(normalizer.std).float()
    wrapped = NormalizedForceTCN(final, mean_t, std_t).eval()
    example = torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    torch.jit.trace(wrapped, example).save(str(args.out / "estimator.ts"))
    torch.onnx.export(wrapped, example, args.out / "estimator.onnx",
                      input_names=["sensor_history"],
                      output_names=["values", "confidence"],
                      dynamic_axes={"sensor_history": {0: "batch"},
                                    "values": {0: "batch"}, "confidence": {0: "batch"}})
    np.savez(args.out / "normalization.npz", mean=normalizer.mean, std=normalizer.std,
             columns=np.array(spec.column_names()) if hasattr(spec, "column_names") else np.array([]))

    # 导出件必须和 eager 一致。上一次就是因为没验这个，拿装错结构的模型做了判断。
    with torch.no_grad():
        probe = torch.randn(16, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
        ev, ec = final((probe - mean_t) / std_t)
        tv, tc = torch.jit.load(str(args.out / "estimator.ts"))(probe)
    parity = max(float((ev - tv).abs().max()), float((ec - tc).abs().max()))

    report = {"init": str(args.init), "real": str(args.real), "dataset": str(root), "holdout": args.holdout,
              "baseline": base, "best": best, "parity_max_abs": parity,
              "epochs": args.epochs, "real_weight": args.real_weight,
              "anchor": {"force_zero": args.anchor_force, "velocity": args.anchor_velocity},
              "sim_rmse_limit": limit, "history": log,
              "elapsed_s": round(time.time() - started, 1)}
    (args.out / "finetune_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'':14}{'起点':>12}{'微调后':>12}{'变化':>12}")
    for key, name, fmt in (("real_p95", "真机误报 p95", "%.2f N"),
                           ("real_p99", "真机误报 p99", "%.2f N"),
                           ("real_max", "真机误报 max", "%.2f N"),
                           ("sim_xy_rmse", "仿真 xy RMSE", "%.3f N"),
                           ("sim_high_rmse", "仿真高力段", "%.3f N"),
                           ("sim_zero_p95", "仿真零力 p95", "%.2f N"),
                           ("real_vel_ratio", "速度头/指令", "%.2f")):
        b, a = base[key], best[key]
        print(f"{name:<14}{fmt % b:>12}{fmt % a:>12}{(a - b) / abs(b) * 100:>11.1f}%")
    print(f"\n[ft] 最佳 epoch {best['epoch']}  TorchScript 一致性 {parity:.2e}")
    print(f"[ft] → {args.out}")


if __name__ == "__main__":
    main()
