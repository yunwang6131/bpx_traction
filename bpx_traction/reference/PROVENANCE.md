# 参考资料来源与用途

收录日期：2026-10-22。本目录是**只读存档**，`bpx_traction` 的代码不 import 它。
收录的目的是：原始交付包会被删除，而牵引这条线的每一条设计决策都应当可追溯到出处。

---

## 1. `core_code/` — 力估计器核心实现代码包

| | |
|---|---|
| 原路径 | `~/intern/ZGYD/力估计器核心实现代码_20261021/force_estimator_core_code_20261021/` |
| 归档 | `~/intern/ZGYD/力估计器核心实现代码_20261021.tar.gz` |
| tar.gz SHA256 | `5cdaf7a41bc58014f06a37606d26229153c58a63a5305f630d868f1de9e223ea` |
| 文件数 | 33（+ `SHA256SUMS.txt`） |
| 完整性 | 全部 33 个文件对原包 `SHA256SUMS.txt` 校验通过，零失败 |
| 来源 | 同事的 Go2 牵引项目（任务 `Unitree-Go2-SelfPull-Gate41-World` / `-Slowdown`） |

**整包收录，不挑拣。** 理由：412K 对仓库可以忽略，而漏收一个文件、原包又已删除，就再也拿不回来了。
下面按"是否已用于 BPX 的设计推导"分类。

### 1.1 已用于推导

| 文件 | 从中取得 |
|---|---|
| `contracts/estimator_io.yaml` | **64 维输入分组的权威定义**（acc3 / gyro3 / proj_g3 / quat4 / q12 / dq12 / τ12 / last_action12 / cmd3）；15 维输出分组；`sample_period_s: 0.02`；`history_steps: 25`；`leakage_policy.forbidden_inputs` |
| `contracts/observation_contract.yaml` | Go2 的 73 维 actor 观测布局；`estimated_force` 在 `[70:73]`、`unit: N_over_80`、`scale: 0.0125`、`frame: body`；证明 Go2 把力**同时**送进速度指令和观测 |
| `contracts/frame_contract.yaml` | Go2 用世界系 W0（`startup_heading_gravity_aligned`）为规范量，机身系是派生量。**BPX 的决定与之相反：直接用机身系**，理由见 `docs/traction_plan.md` |
| `contracts/baseline_manifest.yaml` | 冻结基线 `model_17488`，`force_scale_n: 80`，`policy_force_input_frame: body` |
| `inference/run_estimator_rollouts.py` | **61/64 维帧的实际拼装代码**（第 ~60 行的 `torch.cat`），确认了 `estimator_io.yaml` 的分组顺序；置信度门 `conf_force >= 0.55`；力变化率限幅（`‖ΔF‖>15 N` 且 `‖F_prev‖>1 N` → 只走 35%）；安全闸门（持续低置信度 → `action := 0`）；`p[:,15] = 0`（**wz 恒为零**）；鲁棒注入的实现（噪声/偏置/随机游走/延迟队列/丢包/yaw 漂移）；历史用首帧重复 25 次初始化 |
| `inference/collect_sim_estimator_dataset.py` | 力场景参数的取值范围（见下方 §3）；数据集格式 `{sensor_history, privileged_targets}`；按 env 索引切 70/15/15；**指令通道在估计器输入里被置零**及其理由注释 |
| `training/train_estimator.py` | **完整损失函数**（各项权重、`‖F‖>2` 样本权 ×4、方向项 `1−cos_sim`、置信度标定目标 `exp(−3·err).clamp(0,1)`）；AdamW lr 8e-4 / wd 1e-4 / batch 512 / grad clip 1.0；归一化在 `(0,1)` 维上统计 |
| `verification/analyze_direction_matrix_behavior.py` | **行为判据**：评估窗口 `t≥0.75s`、release 前 0.25s 截止；跟随投影 `≥0.03 m/s`、正向帧占比 `≥0.60`；`‖F_xy‖<8 N` 或纯竖直 → 平均速度 `≤0.15 m/s`；释放后最后 1s `≤0.15 m/s`；高度 `≥0.16 m`、`\|roll\|,\|pitch\| ≤45°` |
| `verification/analyze_play_certification.py` | 系统判据：完成率 ≥95%、CPU p99 <6 ms、ONNX 一致性 <1e-4；鲁棒注入的精确数值（噪声 0.02 / 偏置 0.01 / 随机游走 3e-4 每步 / yaw 漂移 3.0°每场景 / 丢包 0.03 / 延迟 2 步） |
| `inference/analyze_extended_closed_loop.py` | 闭环质量指标：速度跟踪 RMSE、动作变化率 P95、足底打滑率、**支撑多边形裕度**（机身投影到接触足凸包的有符号距离） |
| `acceptance/analyze.py` | 力精度判据（分量 MAE ≤1 N、RMSE ≤1.5 N、幅值 ≤1.5 N、角度均值 ≤5°、P95 ≤10°）；统计口径（每 seed 一个指标、seed 间 Student-t、n=10、Bonferroni 同时置信界） |
| `training/build_state_fusion.py` / `export_state_fusion.py` | 双模型融合的做法（state specialist 接管 `[:4]`，primary 接管其余）。**BPX 第一版不用**，记录备查 |
| `training/assemble_deployment_bundle.py` | 发布状态字段：默认值 `NOT_RELEASED_G4_FAILED_R0_R5_NOT_EXECUTED` |

### 1.2 存档备查（未用于 BPX 设计）

| 文件 | 为什么没用 |
|---|---|
| `training/train_force_estimator_82.py`<br>`training/finetune_force_estimator_82.py`<br>`training/distill_force_estimator_82.py` | 82 维那一代（更早的架构，`ForceEstimator(82,25,32)`，单 head，机身系标签）。微调/蒸馏的加权策略（突变段样本权 ×5、静态段蒸馏权 3.0）日后压精度时可参考 |
| `training/compare_estimator_parity.py` | Go2 的 C++ 部署与 Python ONNX 的一致性比对。BPX 板子是 Python + onnxruntime，不需要跨语言 parity |
| `training/build_estimator_ensemble.py`<br>`training/build_weighted_fusion.py` | 集成/加权融合，"挤最后一点精度"的工具，第一版不用 |
| `verification/analyze_cpp_sim2sim.py`<br>`verification/analyze_continuous_jump_certification.py`<br>`verification/generate_*_report.py` | Go2 专有的分析与报告生成 |
| `diagnostics/analyze.py` / `report.py` | Fz 误差分解。分箱和方向锥的定义可复用（见 §2 的结论） |
| `acceptance/report.py` / `verify.py` | Go2 的 acceptance_v2 报告生成与复核 |
| `inference/collect_targeted.py`<br>`inference/analyze_estimator_loop.py` | 定向数据采集与闭环分析入口 |

### 1.3 原包里**缺失**的模块（import 得到但文件没给）

| 缺的 | 被谁 import | BPX 怎么办 |
|---|---|---|
| `world_force/model.py` → `MissingStateTCN` | 训练 / 导出 / 校验 | 已从权重包的 TorchScript 还原，见 §2 |
| `src/tasks/…` 的 `tether_pull` 事件 | 采集 / rollout | 重写为 IsaacLab event，参数范围见 §3 |
| `runner._follow_command` / `_phase_from_command` | rollout | 重写。BPX 用一阶导纳，见 `bpx_traction/mapping.py` |
| `acceptance_v2_metrics`（`score`/`masks`/`event_metrics`） | `acceptance/analyze.py` | 重写，判据见 §1.1 |
| `direction_balance_common` | `acceptance/analyze.py` | 同上 |
| `experiments/world_force_adapter_v1/config/world_force.yaml` | 打包脚本 | 适配器增益，是 Go2 标定值，BPX 自己标 |
| `experiments/world_force_adapter_v1/config/scenarios.yaml` | rollout | 192 场景方向矩阵，自己生成 |

---

## 2. 已删除的权重包：抢救出来的事实

| | |
|---|---|
| 原路径 | `~/intern/ZGYD/力估计器完整打包_20261021_v2/force_estimator_package_20261021_v2/` |
| 状态 | **2026-10-22 之前已被删除**，`.tar.gz` / `.zip` 一并删除 |

下面这些是在删除前从该包中读出来的。**它们已无法再对原件复核**，记录于此以保留推导链条。

### 2.1 网络结构（从 `models/missing_state_world_v2_estimator.ts` 的 TorchScript 还原）

```
输入 (B, 25, 64) → (x − mean) / std → transpose → (B, 64, 25)
  encoder.0      Conv1d(64 → 64, kernel=1)                    逐帧线性编码
  encoder.2..5   4 × [Conv1d(64 → 64, kernel=3) + Norm + Act]  TCN
  取最后一帧 torch.select(·, 2, -1) → (B, 64)
  values      Linear(64 → 15)
  confidence  Linear(64 → 4) → sigmoid
参数总数 55,315
```

感受野 `1 + 4×(3−1) = 9` 帧 = 180 ms。
**原件未能确认 padding 是否因果**；`bpx_traction/model.py` 明确采用因果（左）padding，
理由：`select(-1)` 取最后一帧，若用 same padding 则最后一帧会看到右侧补的零，
训练与部署不一致。

### 2.2 输出头的激活与量程（同上，从 TorchScript forward 读出 + 极值探测确认）

| 切片 | 名称 | 激活 | 实测范围 |
|---|---|---|---|
| `0:3` | `base_velocity` | `tanh × 2.0` | ±2 m/s |
| `3:4` | `base_height` | `sigmoid × c1 + c2` | 上界 0.150（Go2 的值） |
| `4:8` | `contact_probability` | `sigmoid` | 0–1 |
| `8:12` | `foot_load_latent` | `sigmoid × 2.0` | 0–2 |
| `12:15` | `tether_force` | `tanh × 80.0` | **±80 N** |
| confidence | `[vel, height, contact, force]` | `sigmoid` | 0–1 |

BPX 沿用同样的头结构，但 **height 头量程改为 0.2–0.5 m**
（BPX 站立高度 0.32–0.40 m，见 `bpx_rma` 的 `rewards.base_height.target_height`）。

### 2.3 另一代检查点的元数据（`models/convergence_v1_best_epoch6320.pt`）

```
input_dim      82          （更早的一代，不是 64）
hidden_dim     32
history_length 25
label_contract 'integrated force, current body coordinates'   ← 机身系
结构           encoder.0 + temporal.0/2/4 + 单 head
```

说明 Go2 **早期用机身系标签，后来改成世界系 W0**，原包未记录改动理由。
BPX 的决定是回到机身系。

### 2.4 验收结论（来自 `reports/` 与 `assemble_deployment_bundle.py` 默认值）

```
release_status                  NOT_RELEASED_G4_FAILED_R0_R5_NOT_EXECUTED
formal_application_acceptance   undetermined
default_deployment_changed      False
```

`reports/Fz误差定位报告.md` 的关键数字：

* 全样本描述性拟合 **`pred_Fz ≈ 0.532 × true_Fz − 0.55 N`**（双向幅值系统性低估）
* 向上低估 8.67 N、向下低估 7.91 N，**正负偏差互相抵消**，整体平均偏差仅 −0.50 N，掩盖了问题
* 真实合力 30–45 N 的帧占 26.8%，却贡献 Fz 总平方误差的 **70.9%**
* 报告明确写："不授权直接乘 1/0.532"

对 BPX 的影响：Fz 只估不消费（映射只用 Fx/Fy），该问题自动降级。
损失函数中 Fz 的权重低于 Fx/Fy。

### 2.5 归一化统计量（`configs/missing_state_world_v2_normalization.npz`）

仅含无列名的 `mean` / `std` 各 64 维。从统计量反推分段时发现两点（**现已由
`estimator_io.yaml` 与 `run_estimator_rollouts.py` 独立确认，不再依赖此推断**）：

* 关节位置/速度/last_action 的 std 呈 3 周期（leg-major），而力矩块呈 4 分组
  （type-major，IsaacLab 内部顺序）—— **同一帧里混了两种关节顺序**。
  BPX 一律钉死 `bpx_rma.spec.JOINT_NAMES`（SDK 顺序）。
* 速度指令通道的 std 仅 `[0.139, 0.116, 0.015]` m/s，关节位置 std 仅 0.04–0.06 rad ——
  Go2 的训练数据**几乎是原地被拉**，行走段覆盖很差。
  BPX 的采集改用力 × 方向 × 指令大小 × 指令方向的**正交网格**。

---

## 3. 力场景参数（从 `collect_sim_estimator_dataset.py` 的调用处提取）

`tether_pull` 事件本体未提供，但调用处暴露了全部参数名与取值范围：

```python
duration_s                 = (0.5, 1.5)      cooldown_s = (0.2, 1.0)
static_ramp_release        = True            static_cycle_s = 4.0
static_target_force_range  = (10.0, 40.0)    # N
jump_training              = False           jump_cycle_s = 4.0
jump_pre_s                 = 1.0             jump_post_s  = 1.5
jump_force_range           = (12.0, 40.0)    # N
force_frame                = "world_fixed"
# OOD 集
horizontal_force_range     = (40.0, 55.0)    vertical_force_range = (-28.0, 28.0)
# 定向集
static_fixed_force         = (20.0, 0.0, 0.0)                  # east_hold
jump_fixed_before          = (6.0, -10.39230485, 0.0)          # dir_mag_jump
jump_fixed_after           = (-17.5, 30.31088913, 0.0)
# 接口
set_manual_force(force_w, enable)   last_force_b   last_force_w
```

Go2 整机约 15 kg，BPX-Pro 15.1 kg（手册 9.1），**力的量级可直接沿用**。

采集参数：256 env × 650 步 × stride 10，前 24 步丢弃（历史未填满）。

---

## 4. 本目录不收的东西

| | 为什么 |
|---|---|
| Go2 的 `.onnx` / `.pt` / `.ts` 权重 | 不同机器人的 `M, C, G, J`，权重对 BPX 根本性错误；且原包已删 |
| Go2 的数据集 | 同上 |
| `world_force.yaml`（适配器增益） | Go2 的标定值，BPX 自己标；且原包已删 |
| BPX-Pro 的 USD / MJCF 资产（109M） | `bpx_rma.spec.find_asset_dir()` 既有的外部依赖，不是牵引引入的，收进来会与 `bpx_rma` 重复 |
