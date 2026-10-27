#!/usr/bin/env python3
"""训力估计器。

    conda activate bpx_rma
    python scripts/train_traction.py --epochs 120

不需要 Isaac Sim —— 数据已经采好了，这一步纯 PyTorch。可以和训练并行跑。

## 早停盯的是验收指标，不是总损失

总损失里有七项（速度/高度/触地/负载/力幅值/力方向/置信度标定），而**只有力有消费端**。
总损失降了不代表力估得更准 —— 高度那一项权重是 8.0，它自己变好就能把总损失拉下去。

所以选模型按 **xy 向量误差 RMSE**（映射实际消费的那个量）来挑，总损失只记录不决策。

## 批次按力的大小配平

零力样本占 24%，中高力占一半。随机抽批的话，损失会被"把力估小一点总是安全的"带偏 ——
而零力附近恰恰是日常最常见的工况（机器人大部分时间没被拉），零力误报会让它无故乱走。
配比见 `losses.FORCE_BUCKETS`：锚点 40% / 低力尾部 15% / 中力 15% / 高力 20% / OOD 10%。
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
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bpx_traction import losses, metrics, spec  # noqa: E402
from bpx_traction.augment import ImuAugment, apply_imu_augment  # noqa: E402
from bpx_traction.mapping import AdmittanceParams  # noqa: E402
from bpx_traction import model as model_mod  # noqa: E402
from bpx_traction.model import (  # noqa: E402
    ForceTCN,
    NormalizedForceTCN,
    count_parameters,
    from_checkpoint,
)
from bpx_traction.normalize import Normalizer  # noqa: E402


def load_split(root: Path, name: str):
    data = np.load(root / f"{name}.npz")
    return {k: data[k] for k in data.files}


@torch.no_grad()
def evaluate(model, normalizer, history, targets, device, batch=4096):
    """跑一遍验证/测试集，返回 (预测 values, 置信度)。"""
    model.eval()
    values, confidences = [], []
    mean = torch.from_numpy(normalizer.mean).to(device)
    std = torch.from_numpy(normalizer.std).to(device)
    for begin in range(0, len(history), batch):
        chunk = torch.from_numpy(history[begin:begin + batch]).to(device)
        v, c = model((chunk - mean) / std)
        values.append(v.cpu().numpy())
        confidences.append(c.cpu().numpy())
    return np.concatenate(values), np.concatenate(confidences)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 BPX 牵引力估计器")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据集目录，默认 logs/traction/dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录，默认 logs/traction/estimator")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=512)
    parser.add_argument("--steps-per-epoch", dest="steps_per_epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--cosine", action="store_true",
                        help="学习率余弦退火到 5%%。恒定 lr 后期在最优点附近打摆，"
                             "退火换最后一截精度；默认关，和历史运行保持可比")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--dilations", type=int, nargs="+", default=None,
                        help="每个 TCN 块的空洞率。不给=用模型默认的 1 2 4 8（感受野 620 ms）。"
                             "写 1 1 1 1 退回非空洞的 180 ms，连半个步态周期都不到")
    parser.add_argument("--norm", default="layer", choices=list(model_mod.NORMS),
                        help="归一化方式。group 的统计量跨时间池化，对窗口内的慢漂移"
                             "不敏感 —— 真机误报的主成分正是 0.3–0.9 Hz 慢漂移")
    parser.add_argument("--residual", action="store_true",
                        help="TCN 块加残差连接。空洞带来 gridding 伪影，"
                             "最新帧只有一条通路，残差给它一条直通路")
    parser.add_argument("--patience", type=int, default=25,
                        help="验收指标多少轮没改善就停")
    parser.add_argument("--imu-augment", dest="imu_augment", action="store_true",
                        help="训练时随机缩放/低通加速度的 xy 两轴。仿真的 acc_xy 比真机"
                             "大 6 倍（接触太硬 + 没有柔性），网络会把它当成尺度可信的量；"
                             "真机上读数小 6 倍就被当成'没在动'。见 bpx_traction/augment.py")
    parser.add_argument("--no-balance", dest="no_balance", action="store_true",
                        help="不按力的大小配平批次（消融用）")
    parser.add_argument("--fp-hinge", dest="fp_hinge", type=float, default=0.0,
                        help="零力样本预测幅值超过这个值(N)就线性重罚，对齐误报验收线。"
                             "0=关。建议值 2.0（= 验收线）")
    parser.add_argument("--transient-frac", dest="transient_frac", type=float, default=0.0,
                        help="每批抽这么大比例的瞬态窗样本（力变化后 1 s 内，按 since_change 列选）。"
                             "Go2 的 aligned_window 臂用 64/512=12.5%%。零力误报的大头在松绳后的"
                             "衰减尾巴上，按幅值配平照顾不到这个相位")
    parser.add_argument("--guarded-select", dest="guarded_select", action="store_true",
                        help="选型分 = xy_RMSE × (1 + max(0, 零力FP_p95/2 − 1))：零力误报超过"
                             "2 N 验收线就按超出比例罚。防止'总体赢、零力段退'——a1 那轮"
                             "就是这么回事。默认关，保持和历史可比")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    root = Path(args.dataset) if args.dataset else spec.TRACTION_LOG_DIR / "dataset"
    out = Path(args.output) if args.output else spec.TRACTION_LOG_DIR / "estimator"
    out.mkdir(parents=True, exist_ok=True)

    train = load_split(root, "train")
    valid = load_split(root, "validation")
    normalizer = Normalizer.load(root / "normalization.npz")
    print(f"[traction] 训练 {len(train['history']):,}  验证 {len(valid['history']):,}  "
          f"设备 {device}")

    # 归一化在 GPU 上做，省一次 CPU 往返
    mean = torch.from_numpy(normalizer.mean).to(device)
    std = torch.from_numpy(normalizer.std).to(device)
    train_x = torch.from_numpy(train["history"])
    train_y = torch.from_numpy(train["targets"])

    buckets = losses.bucket_indices(train_y)
    print("[traction] 力桶：" + "  ".join(
        f"{name} {len(idx):,}" for name, idx in buckets.items()))

    transient_pool = None
    if args.transient_frac > 0:
        if "since_change" not in train:
            raise SystemExit("--transient-frac 需要数据集带 since_change 列（新采的才有）")
        pool = np.nonzero(train["since_change"] <= int(1.0 / spec.SAMPLE_PERIOD_S))[0]
        if len(pool) == 0:
            raise SystemExit("since_change ≤ 1 s 的样本是空的 —— 数据有问题")
        transient_pool = torch.from_numpy(pool.astype("int64"))
        print(f"[traction] 瞬态窗池 {len(pool):,}（力变化后 1 s 内），"
              f"每批掺 {args.transient_frac:.0%}")

    dilation_kwargs = {} if args.dilations is None else {"dilations": args.dilations}
    model = ForceTCN(channels=args.channels, blocks=args.blocks,
                     residual=args.residual, norm=args.norm, **dilation_kwargs).to(device)
    cycles = model.gait_cycles_visible()
    print(f"[traction] 参数 {count_parameters(model):,}  空洞率 {model.dilations}\n"
          f"[traction] 感受野 {model.receptive_field()} 帧 = "
          f"{model.receptive_field_s() * 1000:.0f} ms = {cycles:.2f} 个步态周期"
          + ("" if cycles >= 1.0 else "  ⚠️ 不到一个周期，看不到周期性模式"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    # T_max 按名义 epochs；早停时退不到底也没关系 —— 要的只是后段把步长压下来
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        if args.cosine else None)
    weights = losses.LossWeights(fp_hinge_n=args.fp_hinge)
    generator = torch.Generator().manual_seed(args.seed)
    augment = ImuAugment(enabled=args.imu_augment)
    gen_gpu = torch.Generator(device=device).manual_seed(args.seed)
    print(f"[traction] IMU 增广：{augment.describe()}")

    admittance = AdmittanceParams.from_json() if (
        spec.PACKAGE_DIR / "data" / "admittance.json").exists() else AdmittanceParams()
    budget = metrics.command_budget(admittance.virtual_damping_ns_per_m)
    print(f"[traction] 指令预算：xy 力误差 RMSE ≤ {budget:.3f} N"
          f"（B={admittance.virtual_damping_ns_per_m:.2f}，策略跟踪误差 "
          f"{metrics.POLICY_TRACKING_ERROR_MPS:.3f} m/s）")

    best = {"score": float("inf"), "epoch": -1, "xy": float("inf")}
    history_log = []
    since_improved = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_detail = {}
        for _ in range(args.steps_per_epoch):
            if args.no_balance:
                idx = torch.randint(len(train_x), (args.batch_size,), generator=generator)
            else:
                idx = losses.sample_balanced_batch(buckets, args.batch_size, generator)
            if transient_pool is not None:
                n_tr = max(1, int(round(args.batch_size * args.transient_frac)))
                pick = transient_pool[torch.randint(len(transient_pool), (n_tr,),
                                                    generator=generator)]
                idx = torch.cat([idx[: args.batch_size - n_tr], pick])
            raw_x = train_x[idx].to(device)
            raw_x = apply_imu_augment(raw_x, augment, gen_gpu)   # 增广在归一化之前
            x = (raw_x - mean) / std
            y = train_y[idx].to(device)

            values, confidence = model(x)
            loss, detail = losses.estimator_loss(values, confidence, y, weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in detail.items():
                epoch_detail[k] = epoch_detail.get(k, 0.0) + v / args.steps_per_epoch

        if scheduler is not None:
            scheduler.step()

        # 验证：按**验收指标**选模型，不是按总损失
        pred_values, pred_conf = evaluate(model, normalizer, valid["history"],
                                          valid["targets"], device)
        fb, fe = spec.output_slice("force_b")
        report = metrics.force_metrics(pred_values[:, fb:fe], valid["targets"][:, fb:fe])
        score = report["xy_vector_rmse"]
        if args.guarded_select:
            fp95 = report.get("zero_false_positive", {}).get("p95_n", 0.0)
            score = score * (1.0 + max(0.0, fp95 / metrics.Thresholds().zero_false_positive_p95_n - 1.0))

        row = {"epoch": epoch, "score": score, "xy_rmse_n": report["xy_vector_rmse"],
               "zero_fp_p95_n": report.get("zero_false_positive", {}).get("p95_n"),
               "angle_mean_deg": report.get("angle_mean_deg"),
               "mag_rmse_n": report["mag_rmse"], **{f"loss_{k}": v for k, v in epoch_detail.items()}}
        history_log.append(row)

        improved = score < best["score"] - 1e-5
        if improved:
            best = {"score": score, "epoch": epoch, "xy": report["xy_vector_rmse"]}
            since_improved = 0
            torch.save({
                "model": model.state_dict(),
                "channels": args.channels, "blocks": args.blocks,
                "dilations": model.dilations, "residual": model.residual,
                "norm": model.norm_kind,
                "input_dim": spec.FRAME_SIZE,
                "epoch": epoch, "score_xy_rmse_n": report["xy_vector_rmse"],
                "selection_score": score,
            }, out / "estimator.pt")
        else:
            since_improved += 1

        angle = f"{report['angle_mean_deg']:.2f}°" if report.get("angle_mean_deg") else "—"
        fp = report.get("zero_false_positive", {}).get("p95_n")
        print(f"[{epoch:3d}/{args.epochs}] loss {epoch_detail['total']:.4f}   "
              f"xy_RMSE {report['xy_vector_rmse']:.3f} N   零力FP95 "
              f"{fp:.2f} N   角度 {angle}" + (f"   分 {score:.3f}" if args.guarded_select else "")
              + ("  ★" if improved else ""))

        if since_improved >= args.patience:
            print(f"[traction] {args.patience} 轮没改善，停在第 {epoch} 轮")
            break

    # 用最好的那版导出。**必须走 from_checkpoint** —— 手搓 ForceTCN(...) 再 load 的话，
    # 漏传任何一个结构参数都会出事：residual 漏了是**静默**算错（它不带参数，键名一样），
    # norm 漏了会报键名不匹配。两种都在这个文件里真实发生过。
    checkpoint = torch.load(out / "estimator.pt", map_location="cpu", weights_only=False)
    model = from_checkpoint(checkpoint)
    wrapped = NormalizedForceTCN(
        model, torch.from_numpy(normalizer.mean), torch.from_numpy(normalizer.std)
    ).eval()

    example = torch.zeros(1, spec.HISTORY_LENGTH, spec.FRAME_SIZE)
    torch.jit.trace(wrapped, example).save(str(out / "estimator.ts"))
    torch.onnx.export(
        wrapped, example, out / "estimator.onnx",
        input_names=["sensor_history"], output_names=["values", "confidence"],
        dynamic_axes={"sensor_history": {0: "batch"}, "values": {0: "batch"},
                      "confidence": {0: "batch"}},
        opset_version=17,
    )
    normalizer.save(out / "normalization.npz")

    # 导出一致性：板子上跑的必须和训练的是同一个东西
    with torch.no_grad():
        probe = torch.from_numpy(valid["history"][:256])
        eager_v, eager_c = wrapped(probe)
        traced_v, traced_c = torch.jit.load(str(out / "estimator.ts"))(probe)
    parity = max(float((eager_v - traced_v).abs().max()),
                 float((eager_c - traced_c).abs().max()))

    (out / "training_report.json").write_text(json.dumps({
        "best_epoch": best["epoch"],
        "best_xy_rmse_n": best["xy"],
        "best_selection_score": best["score"],
        "command_budget_n": budget,
        "budget_pass": bool(best["score"] <= budget),
        "parameters": count_parameters(model),
        "receptive_field_frames": model.receptive_field(),
        "dilations": model.dilations,
        "norm": model.norm_kind,
        "gait_cycles_visible": round(model.gait_cycles_visible(), 3),
        "torchscript_parity_max_abs": parity,
        "balanced_batches": not args.no_balance,
        "cosine_lr": bool(args.cosine),
        "guarded_select": bool(args.guarded_select),
        "transient_frac": args.transient_frac,
        "fp_hinge_n": args.fp_hinge,
        "dataset": str(root),
        "imu_augment": augment.describe(),
        "loss_weights": vars(weights),
        "elapsed_s": round(time.time() - started, 1),
        "epochs": history_log,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"\n[traction] 最好的是第 {best['epoch']} 轮，xy_RMSE {best['xy']:.3f} N "
          f"（预算 {budget:.3f} N → {'过' if best['xy'] <= budget else '未过'}）")
    print(f"[traction] TorchScript 一致性 {parity:.2e}")
    print(f"[traction] 导出 → {out}")


if __name__ == "__main__":
    main()
