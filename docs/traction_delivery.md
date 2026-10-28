# 牵引层交付说明（v2，2026-10-28）

> 一句话：**仿真侧端到端做完并验收通过，牵引感知层定版可上板。** 剩真机工程集成。

## 1. 它是什么

挂在现有速度策略**外面**的一个上层模块：从本体感知估出外力 → 导纳映射成速度指令
`(vx, vy)` → 喂给现有策略的 command 槽。实现"人牵着机器狗走"。

- **策略零改动**：牵引只写 `velocity_command`，策略不知道牵引存在。
- **底层零改动**：`bpx_rma/` 不碰，45/1125/65 三个观测契约不动（`check_traction_contract.py` 断言）。
- **分层**：L1 估计器（感知）→ L2 导纳映射（决策）→ L3 板端适配（接入）。

## 2. 定版交付物

| 件 | 路径 | 说明 |
|---|---|---|
| 估计器（上板） | `logs/traction/v2/combo_s1_realft/` | 单模型 ts/onnx/归一化。域适配后 |
| 导纳映射参数 | `bpx_traction/mapping.py` `AdmittanceParams` | 死区已标定，`calibrated=True` |
| 板端运行时 | `bpx_traction/board/traction_runtime.py` | 拼帧/推理/映射/门4/失效自关/融合 |
| 契约闸门 | `scripts/check_traction_contract.py` | bpx_rma 零改动 + 帧一致性 |

默认估计器：`spec.DEFAULT_ESTIMATOR_DIR = logs/traction/v2/combo_s1_realft`。

## 3. 关键数字（实测）

**估计器精度**（dataset_v2 测试集 / 真机零力录制）：

| 指标 | 仿真 | 真机 | 备注 |
|---|---|---|---|
| xy 向量 RMSE | 3.98 N | — | 工作区(≤30N)约 2.5N |
| 零力误报 p99 | ~3 N | **16 N** | 域适配后 |
| 方向(15–30N) | 4.8° | — | 过 5° 线 |
| CPU p99 | — | **1.59 ms** | 预算 6ms |

**死区标定**（真机留出段，0.3s 去抖，运动误报 p99=18.6N）：

```
deadzone_on  = 25 N   起步阈值（零误起步的最保守点）
deadzone_off = 12 N   维持阈值（> 运动误报 p50 2.4 / p90 7.9）
B = 13.33, τ = 0.24 s  （力到30N指令到vx上限1.5，反解）
饱和力 = 30 N（设计上限，扛力扫描实测）
```

**闭环四场景**（定版配置，Isaac真实渲染已录视频）：

| 场景 | 结果 | 关键 |
|---|---|---|
| follow | ✓ | 人走11.2m 狗跟11.4m 正向99% |
| turn | ✓ | 跟着拐弯不发散 |
| release | ✓ | 松手 0.36s 停住 |
| idle | ✓ | 没人拉只漂 2cm |

## 4. 核心设计决策（为什么这么做）

- **估计力不估挂点**：输出净力F（机身系），不含挂点/力矩。绳杆通用（力不认来源）。
- **指令通道物理删除**（64→61维）：估计器结构上看不到自己产生的指令，开环训练=闭环部署同分布，不需要 DAgger。
- **标签滞后一步**：观测t体现的是t-1步已积分的力（Go2原始代码环路证明）。
- **融合默认抢占**：手柄一动牵引全让位，wz永远手柄。叠加模式可选（`fusion='add'`），留真机调。
- **仿真训练 + 真机域适配两段式**：仿真底座给精度，真机零力录制压误报。缺一不可——纯仿真模型真机误报 40N+，死区合不拢。

## 5. 剩余工作（真机工程集成）

1. **上板接线**：`traction_runtime.py` 包进手柄脚本 `policy_cmd()` 外层，加 `--traction` 开关（默认关，照门2样式）。
2. **parity 核对**：板上 onnxruntime 输出 vs 训练 torch，< 1e-4。
3. **输入一致性**：运行时必须读**滤波后的 dq**、**clip到±4的 last_action**（和 policy obs 同源，仿真已 assert，板端核对）。
4. **真机行为验收**：一绳一人——拉了跟、松手停、没拉不动。不需测力。

## 6. 已知局限（诚实）

- **精度到物理天花板附近**：工作区 xy 2.5N，距精密测力预算(0.94N)还差；但牵引只需行为对，闭环已证。
- **可用窗口偏窄**（12–30N），控制偏"开关式"，细粒度要更多真机零力数据再压误报。
- **on=25 对纯侧拉偏高**（vy方向饱和力仅19.4N），真机可降 on 调手感。
- **置信度闸门在真机形同虚设**（真机全样本>0.55，含误报），挡误触发靠死区+去抖不靠它。
- **力精度无真机真值**（无配重/拉力计条件），只在仿真验；真机只验行为。

## 7. 复现命令

```bash
conda activate bpx_rma   # 必须 conda activate，直调 bin/python 缺 isaacsim
# 采数据
python scripts/collect_traction.py --headless --num-envs 512 --rounds 16 --stride 10 \
  --policy rollfix=... --policy sprint7=... --output logs/traction/dataset_v2
# 训练（守卫+铰链+幅值匹配）
python scripts/train_traction.py --dataset logs/traction/dataset_v2 --channels 128 \
  --residual --cosine --guarded-select --transient-frac 0.125 --fp-hinge 2.0 --norm-match 0.25
# 集成+蒸馏
python scripts/export_ensemble.py --models s0 s1 s2 --out ensemble
python scripts/distill_ensemble.py --teacher ensemble/estimator.ts --out student
# 真机域适配
python scripts/finetune_real.py --init <底座> --real logs/traction/real_finetune_v2/real.npz \
  --dataset logs/traction/dataset_v2 --out <realft> --holdout 0.3
# 死区标定
python scripts/real_deadzone.py --rec logs/real_v6/rec_20260418_191052 --estimator <realft>
# 闭环+录像
python scripts/play_traction.py --video --video-scene follow --estimator <realft>/estimator.ts
```
