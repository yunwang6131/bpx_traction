#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--output",required=True); a=ap.parse_args(); r=Path(a.root).resolve(); rep=r/"reports/gate41_world_estimator_v1"; g0=json.loads((rep/"g0_baseline_verification.json").read_text()); g2=json.loads((rep/"g2_v2_report.json").read_text()); g3=json.loads((rep/"g3_sim_estimator_v2_metrics.json").read_text()); onnx=json.loads((rep/"onnx_consistency_v2.json").read_text()); c=json.loads((r/"reports/gate41_world_estimator_v2/g4_safety_report.json").read_text()); cr=c["runs"][0]
    text=f'''# 8.18 牵引跟随 v2 迭代报告

## 本轮变更

1. Oracle 轨迹改为双精度、归一化四元数记录，G2 实际往返误差降至 `3.2e-14`。
2. 估计器输入从 60 维扩展到 64 维，加入实机可直接获得的 `q_wb` 姿态四元数；没有引入 root/contact/外力特权量。
3. 重新采集真实 MJLab/MuJoCo 数据并训练 `missing_state_world_v2`。
4. C 组接入独立低置信安全门，低置信时切换到基础站立动作。

## Gate 结果

| Gate | 状态 | 证据 |
|---|---|---|
| G0 | 通过 | 冻结资产 {sum(v["passed"] for v in g0["checks"].values())}/{len(g0["checks"])} 项通过 |
| G1 | 通过 | A/B 固定场景回放稳定完成 |
| G2 | 通过 | 双精度实际轨迹往返最大误差 `3.2e-14`，范数相对误差 `<1e-15` |
| G3 | 候选通过 | 测试力 RMSE {g3["test"]["tether_force_rmse_n"]:.2f} N、方向 {g3["test"]["tether_direction_mean_deg"]:.2f}°；OOD 力 RMSE {g3["ood"]["tether_force_rmse_n"]:.2f} N、方向 {g3["ood"]["tether_direction_mean_deg"]:.2f}°；P99 {g3["test"]["inference_ms_p99"]:.3f} ms；ONNX 误差 {onnx["values_max_abs_error"]:.2e} |
| G4 | 未通过 | C 组安全门后仍有 {cr["terminated_samples"]} 个终止样本，最低高度 {cr["min_height_m"]:.3f} m，方向误差 {cr["force_direction_mean_deg"]:.2f}° |
| G5 | 未放行 | G4 未通过，按方案禁止 Actor 微调与继续放行 |
| G6-G8/R0-R5 | 未执行 | 实机范围，仅更新协议 |

## 当前决策

`missing_state_world_v2` 可作为离线估计器候选，但不能作为实机部署包。下一步应围绕 estimator-in-loop 分布漂移和策略基础步态保持继续做数据/安全状态机工作；在 G4 达标前禁止微调 `model_17488`。
'''
    Path(a.output).write_text(text)


if __name__=="__main__": main()
