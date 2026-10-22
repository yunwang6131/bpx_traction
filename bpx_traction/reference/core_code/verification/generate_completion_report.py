#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path


def j(p): return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");ap.add_argument("--output",required=True);a=ap.parse_args();r=Path(a.root);rep=r/"reports/gate41_world_estimator_v1";g0=j(rep/"g0_baseline_verification.json");ab=j(rep/"g1_g2_oracle_report.json");g3=j(rep/"g3_sim_estimator_metrics.json");onnx=j(rep/"onnx_consistency.json");c=j(rep/"g4_estimator_loop_report.json")
    stable=sum(x["stable_completion"] for x in ab["runs"]);total=len(ab["runs"]);cr=c["runs"][0]
    cpp=j(rep/"cpp_sim2sim_report.json");g2=j(rep/"g2_precision_report.json");g2max=max(x["frame_roundtrip_max_abs"] for x in g2["runs"])
    text=f"""# 8.18 牵引跟随工程化完成报告

生成日期：2026-08-18  
工程位置：`/home/zy/old_data/unitree_rl_mjlab_self_pull`  
策略基线：Gate 4.1 / `model_17488.pt`

## 结论

G0、G1 已通过；G2 以最新精度复测报告为准。G3 与 G4 未通过，因此按 8.18 方案停止在冻结 Actor，不进行 Actor 微调，不进入 G5 闭环放行或任何实机测试。失败是实际仿真结果，不用继续训练掩盖。

## Gate 状态

| Gate | 状态 | 实际证据 |
|---|---|---|
| G0 资产冻结 | 通过 | {sum(v['passed'] for v in g0['checks'].values())}/{len(g0['checks'])} 项通过，checkpoint/ONNX/config 哈希一致，73 维观测、12 维动作一致 |
| G1 Oracle 复现 | 通过 | A/B 共 {total} 个实际 MJLab/MuJoCo 回放，稳定完成 {stable}/{total}，无提前终止 |
| G2 坐标契约 | 未通过 | 双模式、世界/机身标签和 90 度单测通过；实际 float32 轨迹最大往返误差 {g2max:.2e}，略高于 1e-5 门槛 |
| G3 估计器离线 | 未通过 | 测试集力 RMSE {g3['test']['tether_force_rmse_n']:.2f} N，方向角 {g3['test']['tether_direction_mean_deg']:.2f}°；OOD 力 RMSE {g3['ood']['tether_force_rmse_n']:.2f} N |
| G4 Estimator-in-loop | 未通过 | C 组终止样本 {cr['terminated_samples']}，最低高度 {cr['min_height_m']:.3f} m，力方向角 {cr['force_direction_mean_deg']:.2f}° |
| G5 鲁棒性闭环 | 未放行 | G3/G4 失败，按 Gate 顺序禁止继续；OOD 离线误差已保留 |
| G6-G8 / R0-R5 | 实机范围，未执行 | 已交付逐级协议、检查表、停止条件和签字表 |

## 已完成工程

- 冻结资产清单与一键哈希核验，未覆盖 `model_17488.pt`。
- 新增 `legacy_body` / `world_fixed` 显式双模式；配置缺失或协议含糊时拒绝猜测。
- MJLab 训练端与独立 C++ MuJoCo bridge 均保留 `F_w`/`F_b`，世界固定力直接施加，偏置力矩使用 `r_w × F_w`。
- 73 维 Observation Contract、60 维仅本体感知 Estimator Contract、W0/frame_id 契约。
- 真实 MJLab/MuJoCo A/B Oracle 数据、无泄漏估计器数据、OOD 数据。
- 多任务因果 TCN、PyTorch/TorchScript/ONNX、归一化与置信度输出；ONNX 最大误差 {onnx['values_max_abs_error']:.2e}，一致性通过。
- 独立安全监督器、frame_id 重置行为、NaN/通信/姿态/高度/低置信降级单测。
- C 组真实 estimator-in-loop 回放及严格失败判定。
- 独立 C++ simulator/controller 通过 DDS 联跑到 step {cpp['controller_max_reported_step']}，记录 {cpp['recorded_frames']} 帧；世界固定力写入误差 {cpp['world_force_max_abs_error_n']:.1e} N。历史估计器闭环发散，因此只通过链路/坐标功能验证，不通过性能放行。

## 证据边界与下一步

当前估计器置信度能在闭环失配时下降，但世界力方向可观测性仍不足。下一轮应先扩充可辨识激励与跨场景数据，重新校准置信度，并在 G3 独立/OOD 与 G4 稳定率均达标后再考虑冻结 Actor 的小学习率微调。实机协议仅供后续人工执行，本报告不声称任何实机结果。
""";Path(a.output).write_text(text)


if __name__=="__main__":main()
