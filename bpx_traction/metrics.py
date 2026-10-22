"""力估计的验收指标。

口径沿用 Go2 的 `acceptance/analyze.py` 和 `diagnostics/analyze.py`，但判据本身
**按 BPX 自己的数重推**，而且落点不同：

* **方向只看 xy 平面**。映射的方向完全由 `F_xy` 决定，三维角度误差里掺进 Fz 没有意义。
* **零力的方向是未定义的，不是"角度算错了"**。所以角度类指标只在 `‖F_xy‖ ≥ 2 N`
  的样本上算，零力段单独报"误报幅值"。
* **判据折算成指令误差**。力误差要经过导纳的 `1/B` 才变成指令误差，而 B 是设计参数。
  把通过线写成指令误差，B 一改判据自动跟着走，不用重推 —— 见 :func:`command_budget`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec

#: 策略自身的速度跟踪误差，作为力误差预算的参照。
#: 来自 `logs/traction/drag_sweep.json` 的零力基线：rollfix 0.053 m/s（最好的那个基底）。
#: 力估计误差引入的指令误差不该超过策略本来就有的跟踪误差 —— 超过了，牵引就成了
#: 跟踪质量的主要误差源。
POLICY_TRACKING_ERROR_MPS = 0.053


@dataclass
class Thresholds:
    """绝对门槛。沿用参考实现的量级，但 §command_budget 才是真正的依据。"""

    component_mae_n: float = 1.0
    component_rmse_n: float = 1.5
    magnitude_mae_n: float = 1.5
    magnitude_rmse_n: float = 1.5
    angle_mean_deg: float = 5.0
    angle_p95_deg: float = 10.0
    #: 零力段（真值 < 2 N）允许的误报幅值 P95
    zero_false_positive_p95_n: float = 2.0


def command_budget(damping_ns_per_m: float,
                   tracking_error_mps: float = POLICY_TRACKING_ERROR_MPS) -> float:
    """力误差预算 (N)：让它引入的指令误差不超过策略自身的跟踪误差。

    导纳的稳态是 ``v = F / B``，所以 ``Δv = ΔF / B``。要求 ``Δv ≤ tracking_error``
    就是 ``ΔF ≤ B · tracking_error``。

    >>> round(command_budget(17.78), 3)      # B 由 damping_for_saturation(30,…) 定
    0.942
    """
    return damping_ns_per_m * tracking_error_mps


def _angle_deg(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    """xy 平面内的夹角 (度)。调用方负责只传有力的样本。"""
    dot = (pred_xy * true_xy).sum(axis=1)
    norms = np.linalg.norm(pred_xy, axis=1) * np.linalg.norm(true_xy, axis=1)
    cos = np.clip(dot / np.maximum(norms, 1e-9), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def force_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """一组样本的力指标。`pred` / `true` 形状 (N, 3)，单位 N，机身系。"""
    if len(pred) == 0:
        return {"n": 0}

    error = pred - true
    true_mag = np.linalg.norm(true, axis=1)
    pred_mag = np.linalg.norm(pred, axis=1)
    active = true_mag >= spec.FORCE_ACTIVE_N
    xy_active = np.linalg.norm(true[:, :2], axis=1) >= spec.FORCE_ACTIVE_N

    out = {
        "n": int(len(pred)),
        "n_active": int(active.sum()),
        "component_mae": [float(np.abs(error[:, k]).mean()) for k in range(3)],
        "component_rmse": [float(np.sqrt((error[:, k] ** 2).mean())) for k in range(3)],
        "component_bias": [float(error[:, k].mean()) for k in range(3)],
        "mag_mae": float(np.abs(pred_mag - true_mag).mean()),
        "mag_rmse": float(np.sqrt(((pred_mag - true_mag) ** 2).mean())),
        # xy 向量误差 —— 映射真正消费的那个量
        "xy_vector_rmse": float(np.sqrt((np.linalg.norm(error[:, :2], axis=1) ** 2).mean())),
        "xy_vector_p95": float(np.percentile(np.linalg.norm(error[:, :2], axis=1), 95)),
    }

    if xy_active.any():
        angles = _angle_deg(pred[xy_active, :2], true[xy_active, :2])
        out.update({
            "angle_n": int(xy_active.sum()),
            "angle_mean_deg": float(angles.mean()),
            "angle_p95_deg": float(np.percentile(angles, 95)),
        })
    else:
        out.update({"angle_n": 0, "angle_mean_deg": None, "angle_p95_deg": None})

    # 零力段：报误报幅值，不报角度（零力的方向没有定义）
    zero = ~active
    if zero.any():
        fp = pred_mag[zero]
        out["zero_false_positive"] = {
            "n": int(zero.sum()),
            "median_n": float(np.median(fp)),
            "p95_n": float(np.percentile(fp, 95)),
            "max_n": float(fp.max()),
            "above_2n_pct": float((fp > 2.0).mean() * 100),
            "above_5n_pct": float((fp > 5.0).mean() * 100),
        }

    # 幅值的系统性缩放。Go2 那边是 pred ≈ 0.532×true − 0.55，整体低估一半。
    # 单独拟一条线能一眼看出有没有同样的毛病。
    if active.sum() >= 10:
        slope, intercept = np.polyfit(true_mag[active], pred_mag[active], 1)
        out["magnitude_fit"] = {"slope": float(slope), "intercept_n": float(intercept)}

    return out


def stratified(pred: np.ndarray, true: np.ndarray,
               extra: dict[str, np.ndarray] | None = None) -> dict:
    """按力的大小和方向分层。`extra` 里可以再带 policy_id / profile_kind 之类。"""
    true_mag = np.linalg.norm(true, axis=1)
    report = {"overall": force_metrics(pred, true)}

    bins = [(0.0, 2.0, "0–2N"), (2.0, 5.0, "2–5N"), (5.0, 15.0, "5–15N"),
            (15.0, 30.0, "15–30N"), (30.0, 1e9, ">30N")]
    report["by_magnitude"] = {
        label: force_metrics(pred[m], true[m])
        for lo, hi, label in bins
        if (m := (true_mag >= lo) & (true_mag < hi)).any()
    }

    # 方向按八分圆分层 —— 扛力扫描已经显示前后不对称，估计器也可能有方向偏。
    xy_mag = np.linalg.norm(true[:, :2], axis=1)
    live = xy_mag >= spec.FORCE_ACTIVE_N
    if live.any():
        azimuth = (np.degrees(np.arctan2(true[:, 1], true[:, 0])) + 360.0) % 360.0
        octant = (np.floor((azimuth + 22.5) % 360.0 / 45.0) * 45.0).astype(int)
        report["by_direction"] = {
            f"az{a}": force_metrics(pred[m], true[m])
            for a in sorted(np.unique(octant[live]))
            if (m := live & (octant == a)).sum() >= 20
        }

    for key, values in (extra or {}).items():
        report[f"by_{key}"] = {
            str(v): force_metrics(pred[m], true[m])
            for v in np.unique(values)
            if (m := values == v).sum() >= 20
        }
    return report


def confidence_calibration(confidence: np.ndarray, error_n: np.ndarray,
                           bins: int = 10) -> dict:
    """置信度标定：按置信度分桶，看每桶的真实误差。

    映射用 `conf_force` **连续调制斜率**并做硬闸门，所以标定错了比估不准还糟 ——
    高置信度配大误差 = 自信地走错方向。
    """
    order = np.argsort(confidence)
    chunks = np.array_split(order, bins)
    rows = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        rows.append({
            "confidence_mean": float(confidence[chunk].mean()),
            "error_mean_n": float(error_n[chunk].mean()),
            "error_p95_n": float(np.percentile(error_n[chunk], 95)),
            "n": int(len(chunk)),
        })
    # 单调性：置信度越高，误差该越小。用秩相关看方向。
    if len(rows) >= 3:
        c = np.array([r["confidence_mean"] for r in rows])
        e = np.array([r["error_mean_n"] for r in rows])
        rank_corr = float(np.corrcoef(np.argsort(np.argsort(c)),
                                      np.argsort(np.argsort(e)))[0, 1])
    else:
        rank_corr = float("nan")
    return {"bins": rows, "rank_correlation": rank_corr,
            "monotonic": bool(rank_corr < -0.5)}


def verdict(report: dict, budget_n: float, thresholds: Thresholds | None = None) -> dict:
    """对着门槛判一遍。返回每条的通过情况和一句话结论。"""
    t = thresholds or Thresholds()
    overall = report["overall"]
    checks = {
        "component_mae": (max(overall["component_mae"][:2]), t.component_mae_n),
        "component_rmse": (max(overall["component_rmse"][:2]), t.component_rmse_n),
        "mag_mae": (overall["mag_mae"], t.magnitude_mae_n),
        "mag_rmse": (overall["mag_rmse"], t.magnitude_rmse_n),
    }
    if overall.get("angle_mean_deg") is not None:
        checks["angle_mean"] = (overall["angle_mean_deg"], t.angle_mean_deg)
        checks["angle_p95"] = (overall["angle_p95_deg"], t.angle_p95_deg)
    if "zero_false_positive" in overall:
        checks["zero_fp_p95"] = (overall["zero_false_positive"]["p95_n"],
                                 t.zero_false_positive_p95_n)

    rows = {k: {"value": v, "limit": lim, "pass": bool(v <= lim)}
            for k, (v, lim) in checks.items()}
    # 预算判据：xy 向量误差折算成指令误差
    rows["command_budget"] = {
        "value": overall["xy_vector_rmse"],
        "limit": budget_n,
        "pass": bool(overall["xy_vector_rmse"] <= budget_n),
        "note": f"xy 力误差 RMSE ≤ {budget_n:.2f} N ⇔ 指令误差 ≤ "
                f"{POLICY_TRACKING_ERROR_MPS:.3f} m/s（策略自身的跟踪误差）",
    }
    # 注意：只报注意事项，**不组成"应用验收通过"的结论** —— 应用预算（允许多大力误差、
    # 多长恢复时间）还没定下来，这里只能给描述性结果。
    return {
        "checks": rows,
        "absolute_pass": all(r["pass"] for k, r in rows.items() if k != "command_budget"),
        "budget_pass": rows["command_budget"]["pass"],
        "application_acceptance": "undetermined（应用预算未定，这里只做描述性判定）",
    }


def render(report: dict, verdict_rows: dict | None = None) -> str:
    """人看的版本。"""
    lines = []
    o = report["overall"]
    lines.append(f"总体  n={o['n']:,}（有力 {o['n_active']:,}）")
    lines.append(f"  分量 MAE  fx {o['component_mae'][0]:.3f}  fy {o['component_mae'][1]:.3f}  "
                 f"fz {o['component_mae'][2]:.3f}  N")
    lines.append(f"  分量 RMSE fx {o['component_rmse'][0]:.3f}  fy {o['component_rmse'][1]:.3f}  "
                 f"fz {o['component_rmse'][2]:.3f}  N")
    lines.append(f"  分量偏差  fx {o['component_bias'][0]:+.3f}  fy {o['component_bias'][1]:+.3f}  "
                 f"fz {o['component_bias'][2]:+.3f}  N")
    lines.append(f"  幅值 MAE {o['mag_mae']:.3f}  RMSE {o['mag_rmse']:.3f} N")
    lines.append(f"  xy 向量误差 RMSE {o['xy_vector_rmse']:.3f}  P95 {o['xy_vector_p95']:.3f} N"
                 "   ← 映射实际消费的量")
    if o.get("angle_mean_deg") is not None:
        lines.append(f"  xy 方向 均值 {o['angle_mean_deg']:.2f}°  "
                     f"P95 {o['angle_p95_deg']:.2f}°（n={o['angle_n']:,}）")
    if "magnitude_fit" in o:
        f = o["magnitude_fit"]
        flag = "  ⚠️ 系统性低估" if f["slope"] < 0.85 else ""
        lines.append(f"  幅值拟合 pred ≈ {f['slope']:.3f}×true {f['intercept_n']:+.3f} N{flag}")
    if "zero_false_positive" in o:
        z = o["zero_false_positive"]
        lines.append(f"  零力误报 中位 {z['median_n']:.3f}  P95 {z['p95_n']:.3f}  "
                     f"最大 {z['max_n']:.3f} N   >2N 占 {z['above_2n_pct']:.2f}%")

    for group, title in (("by_magnitude", "按力大小"), ("by_direction", "按方向"),
                         ("by_policy_id", "按采集策略")):
        if group not in report:
            continue
        lines.append(f"\n{title}")
        lines.append(f"  {'':<10}{'n':>9}{'xy_RMSE':>10}{'角度均值':>10}{'幅值RMSE':>11}")
        for key, m in report[group].items():
            angle = f"{m['angle_mean_deg']:.2f}°" if m.get("angle_mean_deg") is not None else "—"
            lines.append(f"  {key:<10}{m['n']:>9,}{m['xy_vector_rmse']:>10.3f}"
                         f"{angle:>10}{m['mag_rmse']:>11.3f}")

    if verdict_rows:
        lines.append("\n判据")
        for name, row in verdict_rows["checks"].items():
            mark = "通过" if row["pass"] else "未过"
            lines.append(f"  [{mark}] {name:<16s} {row['value']:.3f} / {row['limit']:.3f}")
        lines.append(f"  绝对门槛 {'全过' if verdict_rows['absolute_pass'] else '有未过项'}；"
                     f"指令预算 {'过' if verdict_rows['budget_pass'] else '未过'}")
        lines.append(f"  应用验收：{verdict_rows['application_acceptance']}")
    return "\n".join(lines)
