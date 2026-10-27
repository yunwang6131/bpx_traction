#!/usr/bin/env python3
"""把集成老师蒸馏成单个学生模型，拿到接近集成的精度但只跑一遍前向。

    conda activate bpx_rma
    python scripts/distill_ensemble.py \
        --teacher logs/traction/v2/ensemble_combo/estimator.ts \
        --dataset logs/traction/dataset_v2 \
        --out logs/traction/v2/student

为什么蒸馏：三模型集成在板上 CPU p99 超 6ms 预算。学生学集成的**输出**（软标签），
比直接学真值标签更好——集成的输出已经把三个种子的噪声平掉了，学生继承这个平滑。

学生 = 和成员同结构的单个 ForceTCN。损失 = 匹配老师的 values + confidence
（不是匹配真值）。这是标准的知识蒸馏。
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from bpx_traction import spec, metrics  # noqa: E402
from bpx_traction.model import ForceTCN, NormalizedForceTCN, count_parameters, from_checkpoint  # noqa: E402
from bpx_traction.normalize import Normalizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="集成 estimator.ts")
    ap.add_argument("--member", default=None,
                    help="任一成员目录，取结构超参（默认猜 teacher 同级的 combo_s1）")
    ap.add_argument("--dataset", default=str(PROJECT/"logs/traction/dataset_v2"))
    ap.add_argument("--out", default=str(PROJECT/"logs/traction/v2/student"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.dataset)
    tr = {k: np.load(root/"train.npz")[k] for k in ("history", "targets")}
    va = {k: np.load(root/"validation.npz")[k] for k in ("history", "targets")}
    normalizer = Normalizer.load(root/"normalization.npz")
    mean = torch.from_numpy(normalizer.mean).to(dev); std = torch.from_numpy(normalizer.std).to(dev)

    teacher = torch.jit.load(args.teacher, map_location=dev).eval()   # 带归一化，吃原始帧
    # 学生：单个 ForceTCN，结构同成员（128ch+残差+空洞）
    mem = Path(args.member) if args.member else Path(args.teacher).parent.parent/"combo_s1"
    sck = torch.load(mem/"estimator.pt", map_location="cpu", weights_only=False)
    student = ForceTCN(input_dim=spec.FRAME_SIZE, channels=sck["channels"], blocks=sck["blocks"],
                       dilations=tuple(sck["dilations"]), residual=bool(sck["residual"]),
                       norm=str(sck["norm"])).to(dev)
    print(f"[distill] 学生 {count_parameters(student):,} 参数（老师是它的 3×）")

    tx = torch.from_numpy(tr["history"])
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.05)
    gen = torch.Generator().manual_seed(args.seed)
    fb, fe = spec.output_slice("force_b")
    best = {"xy": 1e9, "epoch": -1}; since = 0; log = []; t0 = time.time()

    for ep in range(1, args.epochs+1):
        student.train()
        for _ in range(args.steps_per_epoch):
            idx = torch.randint(len(tx), (args.batch_size,), generator=gen)
            raw = tx[idx].to(dev)
            with torch.no_grad():
                t_val, t_conf = teacher(raw)          # 老师软标签（原始帧输入）
            s_val, s_conf = student((raw - mean)/std)  # 学生吃归一化帧
            loss = nn.functional.smooth_l1_loss(s_val, t_val) + 0.5*nn.functional.mse_loss(s_conf, t_conf)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        sched.step()
        # 验收：学生 vs 真值（不是vs老师）—— 要的是真精度
        student.eval(); preds=[]
        with torch.no_grad():
            for b in range(0, len(va["history"]), 4096):
                v,_ = student((torch.from_numpy(va["history"][b:b+4096]).to(dev)-mean)/std)
                preds.append(v.cpu().numpy())
        pv = np.concatenate(preds)
        r = metrics.force_metrics(pv[:,fb:fe], va["targets"][:,fb:fe])
        xy = r["xy_vector_rmse"]; log.append({"epoch":ep,"xy_rmse":xy,"fp_p95":r.get("zero_false_positive",{}).get("p95_n")})
        imp = xy < best["xy"]-1e-5
        if imp:
            best={"xy":xy,"epoch":ep}; since=0
            torch.save({"model":student.state_dict(),"channels":sck["channels"],"blocks":sck["blocks"],
                        "dilations":student.dilations,"residual":student.residual,"norm":student.norm_kind,
                        "input_dim":spec.FRAME_SIZE,"epoch":ep,"score_xy_rmse_n":xy}, Path(args.out)/"estimator.pt") if Path(args.out).exists() or Path(args.out).mkdir(parents=True) or True else None
        else: since+=1
        if ep%5==0 or imp:
            print(f"[{ep:3d}] 学生 xy_RMSE {xy:.3f} N  零力FP95 {log[-1]['fp_p95']:.2f}"+("  ★" if imp else ""))
        if since>=args.patience: print(f"[distill] {args.patience}轮没进步，停"); break

    # 导出学生（带归一化，同其它模型）
    ck = torch.load(Path(args.out)/"estimator.pt", map_location="cpu", weights_only=False)
    model = from_checkpoint(ck)
    wrapped = NormalizedForceTCN(model, torch.from_numpy(normalizer.mean), torch.from_numpy(normalizer.std)).eval()
    ex = torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    torch.jit.trace(wrapped, ex).save(str(Path(args.out)/"estimator.ts"))
    torch.onnx.export(wrapped, ex, Path(args.out)/"estimator.onnx", input_names=["sensor_history"],
                      output_names=["values","confidence"],
                      dynamic_axes={"sensor_history":{0:"batch"},"values":{0:"batch"},"confidence":{0:"batch"}}, opset_version=17)
    normalizer.save(Path(args.out)/"normalization.npz")
    # CPU 延迟
    sing=torch.zeros(1,spec.HISTORY_LENGTH,spec.FRAME_SIZE); ts=torch.jit.load(str(Path(args.out)/"estimator.ts"))
    with torch.no_grad():
        for _ in range(20): ts(sing)
        tt=[]
        for _ in range(200):
            s=time.perf_counter(); ts(sing); tt.append((time.perf_counter()-s)*1000)
    p99=float(np.percentile(tt,99))
    (Path(args.out)/"distill_report.json").write_text(json.dumps({
        "teacher":args.teacher,"best_epoch":best["epoch"],"best_xy_rmse_n":best["xy"],
        "parameters":count_parameters(model),"cpu_p99_ms":p99,"cpu_budget_ms":6.0,"cpu_pass":bool(p99<6),
        "elapsed_s":round(time.time()-t0,1),"epochs":log}, indent=2, ensure_ascii=False))
    print(f"\n[distill] 学生最好 xy_RMSE {best['xy']:.3f} N  CPU p99 {p99:.2f}ms（预算6→{'过' if p99<6 else '未过'}）")
    print(f"[distill] -> {args.out}")


if __name__ == "__main__":
    main()
