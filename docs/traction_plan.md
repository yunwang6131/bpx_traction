# 牵引层迁移方案：Go2 力估计器 → BPX-Pro 速度指令融合

> ## 决策记录（2026-10-22）
>
> 下面的正文是方案推演过程，**部分结论已被讨论推翻**。最终决策以这一段和代码为准，
> 代码里每条决策旁边都写了理由。
>
> ### 已锁死
>
> | | | 落在哪 |
> |---|---|---|
> | 坐标系 | **机身系**，估计器直出 `F_b`，不用世界系 W0 | `spec.FORCE_FRAME` |
> | 输入 | **61 维 × 25 帧**，指令通道物理删除（不是置零） | `spec.TERMS` |
> | `last_action` | 保留。删了它，自主加速会被当成外力 | `spec.TERMS` |
> | 映射 | **一阶导纳** `M·v̇ + B·v = F`，M/B 参数化 | `mapping.AdmittanceMapping` |
> | 输出 | `(vx, vy, 0)`。只消费 Fx/Fy，**wz 不发**，Fz 估但不用 | `mapping.step()` |
> | 增益 | **各向同性**。分轴标到各自上限会把 45° 斜拉扭成 18° | `mapping.step()` |
> | B | 不手填，由 `damping_for_saturation()` 从设计意图反解 | `mapping.AdmittanceParams` |
> | 抗外力 | 保留。它是估计器的信号来源，不是障碍 | — |
> | 底层 | `bpx_rma/` 零改动，由契约闸门断言 | `check_traction_contract.py` |
>
> ### 被推翻的（正文里还留着旧说法）
>
> * ~~"力映射成速度，策略的抗扰会和它抵消"~~ —— **错了**。速度跟踪器不把扰动叠加到
>   指令上，它压制扰动。K 不是辨识量，是设计参数。由此"路线 A / B"那个分叉不存在。
> * ~~"删掉指令通道就解决了闭环循环依赖"~~ —— **说过头了**。删掉的是直接通路；
>   `last_action` 在输入里，而它是策略对指令的响应，间接通路还在。训练时指令和力独立、
>   部署时恒同向，**联合分布变了**。解法是闭环数据回灌，不是可选项。
> * ~~"取消 MuJoCo 依赖"~~ —— 不取消，`pace_bpx_pro` 是项目既有约定。
> * ~~"应用预算参照 Go2"~~ —— 预算从 BPX 自己的数推（见正文 §9）。
>
> ### 新增的认识
>
> * 这就是**导纳控制**（admittance），一阶、无刚度项。和教科书的区别：力是估计的不是
>   测量的（所以外环带宽必须远低于力感知带宽）、内环是学出来的策略不是伺服。
> * 底层正在把**步态相位进观测**（45 → 47 维，`[sinφ, cosφ]` 追加在末尾）。
>   前 45 维一字不动，**`cmd` 槽仍在 `[42:45]`**，牵引接口不受影响。
>   `spec.policy_command_slice()` 动态读，帧再扩也不会指错。
> * `migrate_phase_obs.py` 的权重迁移（第一层插零列、新维初始被忽略、接着训）
>   **直接适用于"力进观测"**：47 → 51 同一套做法。所以那条路**不需要从头重训**，
>   成本比原先估计的低得多。
> * `gait_phase_obs` 在 `‖cmd‖ ≤ 0.1` 时把相位置零。牵引输出连续穿过这个阈值会让
>   相位反复开关 → 映射加了 `min_speed = 0.15` 和 Schmitt 迟滞跳过这个带。
> * 换基底会改变 `last_action` 的分布 → 训练数据**混多个策略的 rollout**，
>   让估计器对此免疫。
>
> ### 仍待定
>
> * **挂点**（`spec.TETHER_OFFSET_B` 现为占位，`TETHER_OFFSET_CONFIRMED = False`）
> * **死区**（等实测走路时的机身力波动）、**K 斜率**（手感调）、**饱和点**（扛力扫描给）
> * 基底：rollfix 横向差，会用 Sprint7 及以后。骨架不依赖基底，可以先做。

状态：**方案推演**。2026-10-21（第二版，基于核心实现代码包重写）。

目标：把 Go2 那条"力估计 → 速度指令"的牵引链路搬到 BPX-Pro，作为一个**只写速度指令**的上层
模块；`bpx_rma/` 和板子的 45 维帧 / ONNX 契约 / PD 内环一个字不改，辨识验证那条线不受影响。

**两个来源包是互补的，都要用：**

| 包 | 有什么 | 没有什么 |
|---|---|---|
| `力估计器完整打包_20261021_v2` | 权重（.ts / .onnx / .pt）、归一化、验收报告、PROTOCOL | 代码 |
| `力估计器核心实现代码_20261021` | 28 个脚本（1691 行）、4 份接口契约 | 权重、数据、6 个关键模块 |

---

## 0. 四条结论

1. **核心代码有了，但仍缺 6 个关键模块**（§1.2）。其中最要命的两个恰好就是你要的东西：
   力→速度的 `_follow_command`，和造牵引力场景的 `tether_pull` 事件。
2. **64 维输入契约现在是确定的，不是推断的**（§1.3），而且在 BPX SDK 上一项不缺、刚好凑满 64。
3. **Go2 的 Gate 4.1 实际上走了两条路**：力既映射成速度指令，**也**直接进 73 维 actor 观测
   （`actor[:,70:73]`）。BPX 的 45 维帧动不了，所以我们只能拿其中一半 —— 这正是你要的那一半，
   但代价要说清楚（§2.1）。
4. **闭环循环依赖是被构造性消除的**：Go2 在采集时把估计器输入里的指令通道**置零**，
   注释写明"so the estimator cannot see its own generated command"。
   这把我上一版担心的 DAgger 需求直接取消了（§3.3）。

---

## 1. 核心代码包的核对结果

### 1.1 包的结构

```
training/      13 个脚本：训练、微调、蒸馏、两种融合、集成、导出校验、打包
inference/      5 个脚本：数据采集、闭环 rollout、闭环分析
verification/   6 个脚本：Torch/ONNX 一致性、方向矩阵认证、Play 认证
acceptance/     3 个脚本：acceptance_v2 的分析 / 复核 / 报告
diagnostics/    2 个脚本：Fz 的力大小 / 方向 / 姿态误差分解
contracts/      4 份 yaml：帧、观测、估计器 I/O、基线清单
```

### 1.2 Go2 包里缺的 6 个模块：**不是阻塞项**

因为是在 BPX 上复现方法、不是跑 Go2 的代码，这 6 个本来每一个都要重写。
缺的只是"Go2 当时填的具体数"，而每一处的**结构和取值范围在调用处都暴露了**：

| 缺的 | 对 BPX 的影响 | 依据在哪 |
|---|---|---|
| `world_force/model.py` → `MissingStateTCN` | 无 | ✅ 已从上一个包的 TorchScript 完整还原（§1.4） |
| `src/tasks/…/tether_pull` 事件 | 无 | 参数名 + 取值范围全在调用处（§3.1）；BPX 要换成 IsaacLab event / MuJoCo `xfrc_applied` |
| `runner._follow_command` | 无 | 签名确定（`F_b/80 (3) → (vx,vy) (2)`，wz 恒 0）；增益本来就要在 BPX 上重标定 |
| `acceptance_v2_metrics` | 无 | 分组 / 掩码 / 事件口径在 `PROTOCOL.md` + `diagnostics/analyze.py` 里写得很细 |
| `world_force.yaml`（适配器增益） | 无 | 阈值都在 rollout 代码里（§2.2）；增益是 BPX 自己的标定量 |
| `scenarios.yaml` | 无 | 字段结构（`id / force_w_n / duration_s / release_at_s / family / release`）和 6 个 id 在认证脚本里；192 场景按 §3.1 的力范围自己生成 |

`_phase_from_command` 直接不要 —— BPX 的 45 维帧里没有步态相位。

### 1.3 64 维输入：契约 + 代码双重确认

`contracts/estimator_io.yaml` 的 `input_groups`，和 `run_estimator_rollouts.py` 里
实际拼帧的代码**逐项一致**：

```python
sensor = cat(
    raw_actor[:, 0:3],    # imu_acc_b           3
    raw_actor[:, 6:9],    # imu_gyro_b          3
    raw_actor[:, 9:12],   # projected_gravity_b 3
    q_input,              # q_wb (wxyz, body→world, 要归一化)  4
    raw_actor[:, 18:30],  # joint_position     12
    raw_actor[:, 30:42],  # joint_velocity     12
    torque,               # motor_torque_estimate 12
    raw_actor[:, 42:54],  # previous_action    12
    raw_actor[:, 13:16],  # velocity_command_b  3
)                         # = 64
```

采样周期 20 ms，历史 25 帧，**index 0 是最新帧**（`torch.roll(history,1,1); history[:,0]=...`）。
历史初始化用**第一帧测量值重复 25 次**，不是零（和 BPX 板子的 `prepare_policy_state` 一样）。

BPX SDK 的覆盖：

| 输入组 | BPX SDK 高频接口 |
|---|---|
| imu_acc_b 3 | `getImuAccHighRate()` |
| imu_gyro_b 3 | `getImuOmegaHighRate()` |
| projected_gravity_b 3 | 由 `getImuQuatHighRate()` 算，板子已有 `projected_gravity_xyzw` |
| q_wb 4 | `getImuQuatHighRate()`（注意 BPX 板子用的是 **xyzw**，契约是 **wxyz**，要转） |
| joint_position 12 | `getJointPositionHighRate()` |
| joint_velocity 12 | `getJointVelocityHighRate()` |
| motor_torque_estimate 12 | `getJointTorqueHighRate()` |
| previous_action 12 | 策略循环里本来就有 |
| velocity_command_b 3 | 见 §3.3，建议**不用** |

**但**板子现在的 `read_high_snapshot()` 只读 q / dq / omega / quat，没读 acc 和 torque。
加两个 getter 会增加每次快照的 SDK 往返，500 Hz 下必须实测耗时。

### 1.4 网络结构（从 TorchScript 还原，可直接重写）

```
输入 (B, 25, 64) → (x-mean)/std → transpose → (B, 64, 25)
encoder.0      Conv1d(64→64, k=1)                 逐帧线性编码
encoder.2..5   4 × [Conv1d(64→64, k=3) + Norm + Act]   TCN
取最后一帧 (B, 64)
values     Linear(64→15)
confidence Linear(64→4) → sigmoid
参数 55,315
```

15 维输出头的激活与量程（`estimator_io.yaml` 的 `output_groups` + TorchScript 实测）：

| 切片 | 名字 | 激活 | 量程 | 帧 |
|---|---|---|---|---|
| 0:3 | `base_velocity_w` | `tanh × 2.0` | ±2 m/s | W0 |
| 3:4 | `base_height` | `sigmoid × c1 + c2` | 上界 0.150 | 重力系 |
| 4:8 | `contact_probability` | `sigmoid` | 0–1 | 足 |
| 8:12 | `foot_load_latent` | `sigmoid × 2.0` | 0–2 | 足 |
| **12:15** | **`tether_force_w`** | **`tanh × 80.0`** | **±80 N** | **W0** |

4 维置信度对应 `[velocity, height, contact, force]`。

### 1.5 坐标系：**当前契约是世界系，不是机身系**（上一版我说反了）

`frame_contract.yaml`：

```
W0 = startup_heading_gravity_aligned      启动时的朝向 + 重力对齐
reset_events: [tether_mode_enable, fall_recovery, estimator_restart]
tether_force_w: canonical = true          ← 规范量
tether_force_b: canonical = false, consumer = frozen_actor_adapter
F_b = R_wb^T · F_w
允许 yaw 漂移 1.0 deg；往返误差 1e-5
```

老的 82 维那一代才是机身系（`convergence_v1` 的 metadata 写着
`label_contract: 'integrated force, current body coordinates'`）。
**Go2 从 body 改成了 world，包里没有写为什么。**

### 1.6 发布状态

`assemble_deployment_bundle.py` 的默认值：

```python
--release-status  default="NOT_RELEASED_G4_FAILED_R0_R5_NOT_EXECUTED"
```

acceptance 的 `decision`：`formal_application_acceptance = 'undetermined'`，
`default_deployment_changed = False`。
诊断报告：`pred_Fz ≈ 0.532 × true_Fz − 0.55 N`，30–45 N 的帧占 26.8% 却贡献 70.9% 的 Fz 平方误差。

→ 搬过来的是**方法、契约和验收口径**，精度要在 BPX 上重新挣。

---

## 2. 融合拓扑：我们只能拿 Gate 4.1 的一半

### 2.1 Go2 实际在做什么

`run_estimator_rollouts.py` 第 84 行，一行里干了五件事：

```python
p[:, 3:6]   = vel_b                                 # 估计的机身线速度 → actor
p[:, 12:13] = height_for_actor                      # 估计的高度 → actor
p[:, 13:15] = runner._follow_command(force_b / 80)  # ★ 力 → 速度指令（vx, vy）
p[:, 15]    = 0                                     # ★ wz 恒为 0
p[:, 16:18] = runner._phase_from_command(p[:,13:15])# 步态相位由指令推出
p[:, 54:58] = contact_for_actor                     # 估计的触地 → actor
# 再往后：
actor = cat(p, force_b / 80)                        # ★ 力本身也直接进观测 [70:73]
```

73 维 actor 观测里，有 **5 组共 23 维**来自估计器。BPX 的 45 维帧里**一组都放不下** ——
放进去就得改帧、改 ONNX 输入维度、重训策略、重做板子校验，也就是碰底层。

所以 BPX 只取打星号的那三条：**力 → (vx, vy)，wz 恒为 0**。这正好是你要的拓扑。

**代价要说清楚**：Go2 的 `model_17488` 是在 `Unitree-Go2-SelfPull-Gate41-*` 任务里训的
（任务名里的 "SelfPull" 就是牵引），它**见过持续外力，而且观测里有力**。
BPX 现有的 student 策略两样都没有，扰动只有 `push_by_setting_velocity`（瞬时速度扰动）。
Go2 那些漂亮的数字（192/192 方向矩阵）是有训练支撑的，BPX 不能直接期待同样的结果。

### 2.2 L2 的参数：现在有真数了，不用猜

从 `run_estimator_rollouts.py` 和认证脚本里抠出来的：

| 环节 | Go2 的值 | 出处 |
|---|---|---|
| 力"激活"阈值 | `‖F‖ > 2 N` | 训练 loss 的 `active`、诊断的 `near_zero` 分组 |
| 置信度门 | `conf_force ≥ 0.55` 才信，否则 `F_b := 0` | rollout 的 `trusted` |
| 力变化率限幅 | `‖ΔF‖ > 15 N` 且 `‖F_prev‖ > 1 N` → 只走 35% 的 delta | rollout 的 `alpha` |
| 安全闸门 | `conf_force < 0.55`（或 `conf_vel` 低）→ `action := 0`（默认站立） | rollout 的 `low_confidence` |
| 跟随阈值 | `‖F_xy‖ ≥ 8 N` 才要求跟着走；`< 8 N` 必须站住 | 方向矩阵认证 |
| wz | 恒为 0 | `p[:,15]=0` |

`_follow_command` 本体没了，但签名是确定的：`(B,3) 的 F_b/80 → (B,2) 的 (vx, vy)`。
BPX 这边按 Go2 的架构图那一格重建，**限幅夹进训练分布**：

```
F̂_b (3) + conf
  → 零偏消除     静止 5 s 取均值，开机标定一次
  → 死区         ‖F_xy‖ < 8 N → 输出 0（照 Go2 的跟随阈值）
  → 置信度门     conf_force < 0.55 → 输出 0
  → 增益 K       N → m/s，vx / vy 各一个，标定得来
  → 变化率限幅   ‖ΔF‖ > 15 N 且 ‖F_prev‖ > 1 N → 只走 35%
  → 一阶低通     τ 可调
  → 限幅         vx ∈ [−1.0, 1.5]，vy ∈ [−0.5, 0.5]   ← bpx_pro_flat 的训练分布
  → wz = 0
```

> 限幅要夹**训练分布**（`bpx_pro_flat` 的 vx ≤ 1.5），不是板子限幅
> （`POLICY_VX_FWD_MAX = 3.0`）。这两个数不一样。

### 2.3 Fz 和坐标系的建议

* **Fz 不进速度映射**，只做置信度和安全闸门（Fz 恰恰是 Go2 系统性低估 0.53× 的那一轴，
  而且 Go2 本来也只用 F_xy 判跟随）。
* **第一阶段用机身系（body），不用 W0。** 理由：消费端（速度指令）本来就是机身系；
  世界系要靠 IMU 的绝对 yaw，BPX 的 yaw 会漂（Go2 的鲁棒测试就专门注入 3.0°/场景的 yaw 漂移）；
  而且 wz 恒为 0，机器人不主动转向，世界系带来的"方向保持"收益在第一阶段用不上。
  如果长场景里方向保持不住，再切 W0 —— 这是个可回退的决定，不是一次性的。

---

## 3. 数据与训练：Go2 的做法可以照抄

### 3.1 力场景生成器（`tether_pull` 事件）

本体没了，但调用处把参数名和取值范围全暴露了：

```python
duration_s = (0.5, 1.5)          cooldown_s = (0.2, 1.0)
static_ramp_release = True       static_cycle_s = 4.0
static_target_force_range = (10.0, 40.0)
jump_training = False            jump_cycle_s = 4.0
jump_pre_s = 1.0                 jump_post_s = 1.5
jump_force_range = (12.0, 40.0)
force_frame = "world_fixed"
# OOD 集：
horizontal_force_range = (40.0, 55.0)   vertical_force_range = (-28.0, 28.0)
# 定向集：
static_fixed_force = (20.0, 0.0, 0.0)                      # east_hold
jump_fixed_before = (6.0, -10.39, 0.0)                     # dir_mag_jump
jump_fixed_after  = (-17.5, 30.31, 0.0)
# 接口：set_manual_force(force_w, enable) / last_force_b / last_force_w
```

Go2 整机约 15 kg，BPX-Pro 15.1 kg —— **这些力的量级可以直接沿用**。

BPX 两条路都走，分工不同：

| | IsaacLab（主力） | MuJoCo / pace_bpx_pro（交叉验证） |
|---|---|---|
| 并行 | 4096 env | 单 env |
| 施力 | `set_external_force_and_torque` 作用在 torso，event term 定义在 `bpx_traction/` | `data.xfrc_applied[torso]`，**只读引用，不改 pace_bpx_pro** |
| 物理 | 辨识 yaml（和策略同一套） | `corrected` 模型 |
| 用途 | 训练集 + 消融 | 换引擎复核，绕开"用辨识模型评辨识"的循环论证 |

MuJoCo 那条直接复用 `scripts/sim2sim.py` 已经搭好的通路。

### 3.2 数据集格式与切分

```
npz: {sensor_history: (N, 25, 64) float32, privileged_targets: (N, 15) float32}
按 env 索引切 70 / 15 / 15，相邻窗口留在同一份里（防泄漏）
额外两份：ood.npz（高力段）、jump.npz（突变段，stride 5 而不是 10）
manifest.json 记 seed / 力场景帧 / 形状 / 泄漏声明
```

采集参数：256 env × 650 步 × stride 10，前 24 步丢掉（历史没填满）。

`estimator_io.yaml` 的 `leakage_policy.forbidden_inputs` 直接照抄：
`root_state, contact_sensor_truth, external_force_generator, tether_force_w,
base_velocity_w, base_height_truth`。

### 3.3 指令通道：置零，循环依赖就没了

`collect_sim_estimator_dataset.py`：

```python
proprio[:, 13:16] = 0.0
# Zero command entries exactly as deployment does so the estimator
# cannot see its own generated command.
```

**这把闭环分布偏移从根上解决了** —— 估计器看不到自己产生的指令，训练（开环）和部署（闭环）
在输入上是同一个分布。我上一版方案里排的 DAgger 那一轮可以取消。

> ⚠️ **两个脚本不一致，要当心**：`collect_sim_estimator_dataset.py` 置零了，
> 但 `run_estimator_rollouts.py` 喂的是 `raw_actor[:,13:16]`（真实指令，没置零）。
> 而归一化 npz 里指令通道的 std 是 0.139 / 0.116 / 0.015 —— 不是 0，也不大。
> 这说明 64 维那一代的训练数据既没有完全置零、也没有充分覆盖指令分布。
>
> **BPX 的做法：把这 3 维从估计器输入里物理删掉（64 → 61 维）**，并在契约里写明理由。
> 比"约定置零"更不容易出错 —— 置零是个靠人守的约定，删掉是结构保证。

类似地，还有一个**力矩顺序不一致**：采集脚本对力矩做了重排
`[3,4,5,0,1,2,9,10,11,6,7,8]`，rollout 脚本直接用 `actuator_force` 没重排。
这两个脚本服务的是不同代的模型（82 维 / 64 维），但 64 维那一代的训练数据生成器不在包里，
所以**它的力矩顺序没有任何一处可以核对**。BPX 一律钉死 `spec.JOINT_NAMES`（SDK 顺序），
并在契约校验脚本里断言。

### 3.4 损失函数（完整给出，可直接移植）

```python
loss = 1.0 * smooth_l1(vel)              # velocity_weight
     + 8.0 * smooth_l1(height)           # height_weight，权重很高
     + 1.0 * BCE(contact)
     + 1.0 * smooth_l1(load)
     + 3.0 * force                       # smooth_l1(F/40)，‖F_true‖>2 的样本权重 ×4
     + 1.5 * direction                   # 1 − cos_sim，只在 ‖F_true‖>2 的样本上
     + 0.5 * calibration                 # MSE(conf, exp(−3·error).clamp(0,1))

AdamW lr 8e-4, wd 1e-4, batch 512, grad clip 1.0
归一化：mean/std 在训练集的 (0,1) 维上统计
```

**BPX 上只有 force 头有消费端**（其余四组没地方放，45 维帧动不了）。
但**四个辅助头要保留训练** —— 它们是共享主干的正则项，Go2 给 height 的权重高达 8.0，
删掉会直接改变主干学到的表示。只是推理时不消费而已。

微调 / 蒸馏那套（`finetune_*` / `distill_*`：突变段样本权重 ×5、静态段蒸馏权重 3.0、
lr 5e-5、grad clip 0.5）是"挤最后一点精度"的工具，BPX 第一轮不用，留在后面。

---

## 4. 验收：Go2 的判据可以直接搬

### 4.1 行为判据（`analyze_direction_matrix_behavior.py`）

这是**牵引行为本身**的判据，不是力精度的判据 —— 正是 BPX 最需要的那一层：

| 检查 | 阈值 |
|---|---|
| 评估窗口 | `t ≥ 0.75 s`（避开起始瞬态），release 前 0.25 s 截止 |
| 跟随方向 | 速度在请求力方向上的投影 `≥ 0.03 m/s` |
| 跟随一致性 | 投影为正的帧占比 `≥ 0.60` |
| 不该走的时候站住 | `‖F_xy‖ < 8 N` 或纯竖直 → 平均速度 `≤ 0.15 m/s` |
| 释放后停住 | 最后 1 s 平均水平速度 `≤ 0.15 m/s` |
| 没摔 | 不终止；高度 `≥ 0.16 m`（Go2 的数，BPX 换成自己的）；`|roll|,|pitch| ≤ 45°` |
| 规模 | 192 场景 × 多 seed，每个 seed 要 192/192 全过 |

### 4.2 系统判据（`analyze_play_certification.py`）

```
完成率 ≥ 95%（nominal 和 robust 各一套）
无系统性场景失败（同一场景挂 ≥2 次）
ONNX 一致性 < 1e-4
CPU 推理 p99 < 6 ms
鲁棒注入（精确值，要逐项核对）：
  sensor_noise 0.02 / sensor_bias 0.01 / random_walk 3e-4 每步
  yaw_drift 3.0°每场景 / dropout 0.03 / delay 2 步
```

鲁棒注入的实现就在 `run_estimator_rollouts.py` 里（相对尺度噪声、常值偏置、随机游走、
延迟队列、丢包用上一帧顶、yaw 漂移直接改四元数再归一化），可以整段照抄。

### 4.3 闭环质量判据（`analyze_extended_closed_loop.py`）

速度跟踪 RMSE、动作变化率 P95、足底打滑率（接触帧里水平足速 > 0.1 m/s 的比例）、
**支撑多边形裕度**（机身投影到接触足凸包的有符号距离，负 = 在外面或接触足少于三个）。
这几个指标和 BPX 现有的 `play.py --report` 可以合并。

### 4.4 力精度判据（`acceptance/`）

分量 MAE ≤ 1 N、分量 RMSE ≤ 1.5 N、幅值 MAE/RMSE ≤ 1.5 N、
角度均值 ≤ 5°、角度 P95 ≤ 10°（只在真值 ≥ 2 N 上算）。
统计口径：**每个 seed 一个指标，seed 之间做 Student-t，n = 10**；
100 个继承测试做 Bonferroni 同时置信界。

> Go2 在这套口径下的结论是 `formal_application_acceptance = 'undetermined'` ——
> 因为应用预算（允许多大力误差、多长恢复时间）一直没定。
> **BPX 如果也不定，P2 同样只能出描述性报告。** 见 §7。

---

## 5. 分层设计

```
L0  底层（不碰）     bpx_rma/ 全部 · 板子的 45 维帧 / ONNX / PD · 辨识对照那四个策略
                     ──────── 以上任何文件不新增、不修改 ────────
L1  感知层           bpx_traction/estimator/   25×61 → 15 + 4
L2  决策层           bpx_traction/mapping/     F̂_b, conf → (vx, vy, 0)   纯函数，可单测
L3  接入层           bpx_traction/adapters/    isaac / mujoco / board 三个入口
                                               只负责把 cmd 递给原有的 command 槽
```

L0 的保护写进验收脚本：

* 新代码一律在 `bpx_traction/`；`git diff --stat bpx_rma/` 必须为空；
* 新 event / observation 在 `bpx_traction/` 里定义，用 cfg 组合进去，不改 `bpx_rma/mdp/`；
* 训练与日志走独立实验名（`bpx_traction_*`）和独立 logs 子树；
* 板子脚本不改 `build_obs_frame` / `run_policy` / `make_policy_session`，
  只在 `policy_cmd()` **外面**包一层，默认关闭。

分层切点是唯一的：板子上 `policy_cmd()` → `build_obs_frame(command, ...)`。
命令是单一来源，帧和历史都从它来，所以牵引层只写这 3 个数就够了。

---

## 6. 分期

| 期 | 内容 | 产出 | GPU |
|---|---|---|---|
| **P0** | 还原 `MissingStateTCN`（§1.4）；写 `bpx_traction/spec.py` 钉死 61 维分段 / 25 帧 / 20 ms / ±80 N / SDK 关节顺序 / 机身系；L2 纯函数 + 单测（给一条力序列，断言指令序列）；`check_traction_contract.py`（含 "bpx_rma 零改动" 闸门） | 能跑的空壳 + 单测全绿 | 否 |
| **P1** | 重写 `tether_pull`（IsaacLab event + MuJoCo xfrc）；按 §3.1 的参数生成力场景；采 train/val/test/ood/jump 五份 | 数据集 + manifest | 是 |
| **P2** | 训练（§3.4 的损失原样搬）；离线验收按 §4.4 的口径出报告 | v0 模型 + 报告 | 是 |
| **P3** | 仿真闭环：`play_traction.py`；§4.1 的方向矩阵 + §4.2 的鲁棒注入 + §4.3 的闭环质量 | 认证报告 | 是 |
| **P4** | 上板（默认关）：板子加 acc / torque 读取（先实测 500 Hz 耗时）+ `--traction` 开关 + **门 3**（默认 `false`，照门 2 的样子）；Torch vs 板上 onnxruntime 的 parity | 真机记录 | 否 |

P0 现在就能做，完全不影响正在跑的训练。

---

## 7. 开工前提核对：代码侧不缺东西

动手前该确认的前提，逐条在本机验过：

| 前提 | 结论 | 怎么验的 |
|---|---|---|
| MuJoCo 能不能加外力而不改 `pace_bpx_pro` | ✅ 可以 | `FloorSim.step()` 每步只清 `d.qfrc_applied`，**不清 `d.xfrc_applied`**。从外面写 torso 的 6 维 wrench 就能持续生效 |
| IsaacLab 有没有加速度计 | ✅ 有 | `isaaclab.sensors.ImuCfg` + `mdp.imu_lin_acc`；scene 里加一个 Imu sensor 即可 |
| 仿真加速度含不含重力 | ✅ 含 | `ImuCfg.gravity_bias` 默认 `(0,0,9.81)`，`lin_acc_w = Δv/dt + bias` —— 和真 IMU 一致，也和 Go2 归一化里 `mean[2] = 9.87` 对得上。**这一条不能搞错，否则 sim2real 直接废** |
| 触地 / 足底力标签 | ✅ 有 | `ContactSensorCfg` 已覆盖全身，足部已用于 air time 奖励 |
| 闭环要用的策略 | ✅ 在 | `logs/rsl_rl/bpx_pro_flat/2026-10-10_11-42-30_student/exported/policy.onnx` |
| BPX 站立高度（height 头的量程） | ✅ 0.40 m（flat）/ 0.32（fast）/ 0.37 | `rewards.base_height.target_height`。**不是 Go2 的 0.150**，量程要重定 |
| 力的量级能不能照搬 | ✅ 能 | Go2 约 15 kg，BPX-Pro 15.1 kg；10–40 N 静态、40–55 N OOD 直接沿用 |

### 7.1 二次开发手册 V1.0.3 查到的（和没查到的）

**✅ IMU 坐标系与身体坐标系平行，无旋转变换**（手册 2.3.3）。
加速度计的轴就是机身轴，不用标安装旋转。这一条省掉一整个标定环节。

**⚠️ 四元数顺序：手册和部署代码直接冲突。**

| 来源 | 顺序 |
|---|---|
| 手册 6.3 `getImuQuat(float[4])` | 明确标注 **(w,x,y,z)** |
| 手册 8.5 `getImuQuatHighRate(float[4])` | **没标顺序** |
| 板子 `bpx_dual_pad_flush.py` | `projected_gravity_xyzw()` 按 **xyzw** 解，喂的是 HighRate |

板子上策略能跑（投影重力错了会立刻摔），说明 HighRate 实际是 xyzw。
但这件事**没有任何文档支撑，而且门 2 不覆盖它**（门 2 管的是关节方向/零位/顺序）。
`q_wb` 占估计器输入 4 维，投影重力还由它导出 —— 共 7 维取决于这个顺序。
**P0 要把它钉进 `bpx_traction/spec.py`，P4 上板前实测确认。**

**⚠️ 关节符号：手册的运动范围和训练用的约定相反。**

| 关节 | 手册 9.2 | `spec.JOINT_POS_LIMITS` |
|---|---|---|
| 侧摆 HipX | −0.87 ~ +0.87 | (−0.8727, 0.8727) ✅ 一致 |
| 髋 HipY | −2.60 ~ +0.79 | (−0.7854, **2.60661**) ⚠️ 整体取反 |
| 膝 Knee | **+0.37 ~ +2.57** | (**−2.7531**, −0.5539) ⚠️ 整体取反 |

手册写"关节零位对应机器人**趴下**姿态"，而策略的 `q0 = [0, 0.7, −1.4]`。
板子实测能跑 ⇒ SDK 实际返回的是训练那一套约定，**手册 9.2 那张表不能直接用**。
这正是门 2 存在的理由。对估计器的影响：关节位置、速度、**力矩** 三组共 36 维里有 24 维
（髋 + 膝）会整体变号 —— 一律钉死 `spec.JOINT_NAMES` 和板子实测约定，不碰手册那张表。

**❌ `getJointTorqueHighRate()` 的语义：手册查不到。**
手册 8.5 只给了"最新高频关节力矩 (N·m)"，没说是测量值还是电流估算、没有滤波/延迟说明。
不用再翻手册了，这一条只能在真机上标。**标定方法**（两条，互相校验）：

1. **静态重力法**：机器人站住不动，已知姿态下用 URDF 算每个关节的重力力矩，
   和 `getJointTorqueHighRate()` 读数比 —— 同时验出符号、单位和零偏。
2. **PD 反推法**：关节模式下发已知 `kp / q_cmd`，读 `q`，比 `kp·(q_cmd − q) − kd·dq`
   和读数 —— 验增益和延迟（延迟用阶跃响应的相位差读）。

在这之前，板子的 500 Hz 记录器只记 `tff`（策略模式下恒为 0），**历史记录里没有一帧真实力矩**，
离线核对不了。力矩占估计器输入 12 维，是 61 维里唯一完全没验过的一组。

### 7.2 还没落实的两条（真机侧，不挡 P0–P3）

1. **板子读 acc + torque 的耗时没实测。** 500 Hz 下每次快照多两个 SDK 往返。
2. **真机上没有力的真值。** 要在真机验力精度，牵引绳上得串一个拉力计 —— 硬件需求。
   不串的话真机只能验行为（跟不跟随、释放后停不停），验不了力。
   手册 9.1：最大有效载荷 8 kg（≈78 N），所以 0–45 N 的牵引力在机器人承受范围内。

## 8. 已定的两件事

1. **坐标系 = 机身系（body）。** 估计器直接输出 `F_b`，不设 W0、不做世界系往返、
   不需要跟踪绝对 yaw。消费端（速度指令）本来就是机身系。
2. **指令通道从输入里删掉 → 61 维。** 单帧 = `acc 3 + gyro 3 + proj_g 3 + quat 4 +
   q 12 + dq 12 + τ 12 + last_action 12`。估计器结构上看不到自己产生的指令，
   训练（开环）和部署（闭环）输入同分布，不需要 DAgger。

这两条直接写进 `bpx_traction/spec.py`，由 `check_traction_contract.py` 断言。

## 9. 应用预算：用 BPX 自己的数推一个，等你确认

预算不定，P2 只能出描述性报告、出不了"通过/不通过"。下面是从 BPX 自己的测量值推的提案。

### 9.1 力误差预算 ≈ 1.1 N（分量）

```
映射满量程   死区 8 N → F_max 40 N 映射到 vx 0 → 1.5 m/s
             （1.5 = bpx_pro_flat 的训练分布上限；40 N 处水平力/重力 = 0.27）
线性增益     K = 1.5 / (40 − 8) = 0.047 m/s / N
力误差传导   分量误差 1 N → 指令误差 0.047 m/s

策略自身跟踪误差（logs/sim2sim/summary.json，MuJoCo corrected）
  指令 1.5 m/s → lin_err 0.051 m/s
  指令 2.0 m/s → lin_err 0.058 m/s
  指令 2.5 m/s → lin_err 0.073 m/s

判据：力估计误差引入的指令误差不超过策略自身的跟踪误差
  → 分量误差预算 = 0.051 / 0.047 ≈ 1.1 N
```

取整提案：**分量 MAE ≤ 1.0 N，分量 RMSE ≤ 1.5 N，幅值 MAE/RMSE ≤ 1.5 N**；
方向（真值 ≥ 2 N 时）**角度均值 ≤ 5°，P95 ≤ 10°**。

> 角度预算的来源：8 N 处方向错 10° 的横向分量 = 8·sin10° = 1.4 N，
> 乘 K 后是 0.065 m/s 的横向指令误差 —— 和纵向预算同量级，自洽。

### 9.2 响应时间预算

| 环节 | 数 | 来源 |
|---|---|---|
| TCN 感受野 | 9 帧 = **180 ms** | `1 + 4×(3−1)`，结构决定的下限 |
| 历史窗口 | 25 帧 = 500 ms | 契约 |
| 控制周期 | 20 ms | 50 Hz |

提案：**力阶跃 → 指令跟上 ≤ 0.5 s**（约 2.8 倍感受野，留足滤波和限幅的余量）；
**释放 → 停住 ≤ 1.0 s**，判据为最后 1 s 平均水平速度 ≤ 0.15 m/s。

### 9.3 指令抖动预算

提案：牵引层输出的 `vx/vy` 变化率 **P95 ≤ 2.0 m/s²**。
依据：`ACTION_SCALE = 0.1875`、50 Hz、`ACTION_CLIP = ±4`，策略单步能动的关节目标是
±0.75 rad；指令抖动超过这个量级会直接打到动作限幅上。P3 要用实测的动作变化率 P95 复核。

**这三组数需要你确认或改。** 确认之后它们就是 P2/P3 的通过线，不再是"描述性报告"。

## 10. 还要决定的一件事

**第二阶段**（带持续外力重训底层，独立实验名如 `bpx_pro_flat_traction`）要不要现在预留算力？

建议等 P3 的数出来再定。现有 student 策略的扰动训练只有 `push_by_setting_velocity`
（瞬时速度扰动），没见过持续外力。40 N 持续力在 15.1 kg 上是 2.6 m/s² 的恒定加速度扰动 ——
能不能扛住，P3 一跑就知道，现在猜没意义。
