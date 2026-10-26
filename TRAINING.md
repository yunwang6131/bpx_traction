# BPX-Pro 策略训练配方（可复现 / 可续 / 可做对照组）

本文把 sprint7（当前最佳：极速 ~4.5、横滚修复、抬脚 7cm、45 维不碰板子）的训练过程
写成**不依赖任何人记忆**的配方。目标三个：①别人照着能复现；②以后想加第五步能顺着
同一框架续；③换参数 yaml 就能跑无辨识对照组。

> 核实日期 2026-10-22。所有迭代数、种子、cfg 名都从代码和训练日志里核对过，不是凭记忆。

---

## 0. 一句话结构

训练是**四步课程（curriculum）**：由易到难，每步在上一步的 checkpoint 上微调、只加一个
新目标。不是一次从零训出来的——因为抬脚、能量门控这些目标**从零训练会卡在起飞期**
（策略学会消极站桩/慢腾挪拿低分，再也不往"好好走"探索），必须从"已经会走"的
checkpoint 续训才绕得过去。

```
① 基础      sprint3 从零         会走、跟速度、站稳
② 抬脚+横滚  Sprint4 续训         摆动腿抬到 7cm + 机身不左右歪
③ 能量门控  Sprint7 续训         高速敢发力，极速 3.8→4.5
④ 蒸馏导出  student + ONNX       上板版（45 维）
```

**"起飞（takeoff）"** = 策略从乱动突然学会协调行走的转折点，跟踪奖励在几百迭代内从
0.3 冲到 1.0+。卡起飞期 = 奖励一直趴在 0.3-0.4 上不去。从零训 sprint3 在 1100-1400
迭代间起飞；后面几步从种子续训，起点已在起飞之后，不用赌这个转折。

---

## 1. ⚠️ 当前代码状态 ≠ 历史训练路径（必读）

代码在开发中演进过，**后面步骤的改动被合进了前面的 cfg**。所以"现在的某个 cfg"和
"当年训练时的那个 cfg"不完全一样。核对过的现状：

| cfg（velocity_env_cfg.py） | 现在包含什么 | 和历史的差异 |
|---|---|---|
| `BpxProFlatSprint3EnvCfg` | 基础（跟踪/姿态/站立/抬脚时长/辨识 DR） | 与历史一致 |
| `BpxProFlatSprint4EnvCfg` | sprint3 + **抬脚 deficit** + **flat_roll 横滚** | ⚠️ 横滚是 10-21 才加的，当年 10-19 训抬脚时**没有** flat_roll |
| `BpxProFlatSprint7EnvCfg` | sprint4 + **能量门控** | 与历史一致 |

**含义**：当前的 `Sprint4-v0` 一次训练会**同时**学抬脚+横滚，而历史上是**先抬脚
（10-19）、再横滚（10-21）分两步**。一次学两个未单独验证过，可能行、可能相互干扰。
所以下面给**两种复现模式**。

---

## 2. 两种复现模式

### 模式 A：从零一键跑完整链（完全独立复现，但有风险）

用当前固化的 cfg，从零走到底。适合"想完全不依赖任何已存 checkpoint 复现"的人。

```bash
# 辨识组
bash scripts/chain_full.sh identified
# 无辨识对照组（唯一区别：参数 yaml）
bash scripts/chain_full.sh baseline
```

**风险**：① sprint3 从零有起飞随机性（不同随机种子起飞点不同，极少数可能不起飞）；
② Sprint4 一次学抬脚+横滚未验证。如果某步 play 复测不达标，退回模式 B 分步做。

### 模式 B：从种子 checkpoint 分步接力（稳，推荐做增量调整）

每步从上一步**已保存的** teacher checkpoint 续训。这是实际走过、验证过的路径，最稳。
已有的种子 checkpoint（`released/` 和 `logs/rsl_rl/` 里）：

| 步 | 种子 checkpoint | 产物 |
|---|---|---|
| ① | 无（从零） | sprint3 teacher `model_4199` |
| ② | sprint3 `model_4199` | sprint4(含横滚) teacher `model_7056`（即 rollfix） |
| ③ | rollfix `model_7056` | sprint7 teacher `model_8255` |
| ④ | sprint7 `model_8255` | sprint7 student + ONNX |

做增量调整（比如加第五步）时用模式 B，从最近的 teacher 续训。

---

## 3. 四步详解（目的 / cfg / 迭代 / 切换标准 / 复测）

### ① 基础：sprint3 从零

- **目的**：学会走、跟踪速度指令、站稳。这是整条链的地基。
- **task**：`Isaac-Velocity-Flat-BpxPro-Sprint3-v0`
- **种子**：无，从零
- **迭代**：4200（teacher）
- **切换标准（达标才进下一步）**：跟踪奖励 `track_lin_vel_xy_exp` **起飞并稳定在 1.0+
  不再涨**（看 tensorboard 曲线，起飞后再训 500+ 迭代确认稳住）。
- **复测**：`play.py --task ...Sprint3-Play-v0 --vx 1.5 --report`，跟踪误差应 <0.1。

### ② 抬脚 + 横滚：Sprint4 续训

- **目的**：摆动腿抬到 ~7cm（不蹭地）+ 机身横滚压到 <1°（不左右歪）。
- **task**：`Isaac-Velocity-Flat-BpxPro-Sprint4-v0`（当前 cfg 已含抬脚 deficit + flat_roll）
- **种子**：sprint3 `model_4199`
- **迭代**：模式 A 一次约 1500；模式 B 历史是分两步（抬脚 ~1200 + 横滚 400）
- **抬脚用的是线性 deficit**（`foot_clearance_deficit`，cfg 默认 `lin:-3.0`）。
  ⚠️ **开发弯路（别踩）**：最初用二次型 `foot_clearance_penalty`（-2、-6）抬脚卡在
  2.5cm 平台，换成线性 deficit 才达标。配方用线性，二次型是失败教训、不要复现。
- **切换标准**：play 复测 `foot_apex` 逐摆动真峰值 **≥6cm** **且** 横滚 **<1°** **且**
  跟踪没掉（<0.12）。三个同时满足才算这步成。
- **复测**：`play.py --task ...Sprint4-Play-v0 --agent rsl_rl_cfg_entry_point
  --vx 1.5 --report`，看 `foot_apex` / `机身倾斜(横滚)` / `lin_err`。

### ③ 能量门控：Sprint7 续训

- **目的**：破极速的"站桩天花板"。低速约束不变，高速放松力矩/加速度/动作率惩罚，
  让策略高速敢发力。极速 3.8 → 4.5。
- **task**：`Isaac-Velocity-Flat-BpxPro-Sprint7-v0`
- **种子**：rollfix `model_7056`（即步②的产物）
- **迭代**：1200
- **门控参数**（cfg 里 `GATE`）：低速(≤2.0)全额罚，高速(≥4.5)降到 10-20%，指令上限 5.0。
  ⚠️ **开发弯路**：sprint8 把门控旋钮开到头（gate_min 0.03、v_hi 5.5、指令 6.0）
  极速没突破 4.1、6.0 指令反而退化——确认 **4.1-4.5 是 33N·m 电机的物理天花板**，
  再开门控没用。所以门控用 sprint7 的档位就够，别学 sprint8。
- **切换标准**：斜坡升速实际极速 **≥4.3**（`play.py --vx 5.0 --ramp 0.7 --report`
  看 `vx_actual`）**且** 低速（≤2.0）跟踪/横滚和步②一样没退化。
- **复测**：`play.py --task ...Sprint7-Play-v0 --vx 5.0 --ramp 0.7 --report`。

### ④ 蒸馏 + 导出

- **目的**：teacher（看特权观测）→ student（只看 45 维历史，可上板）→ ONNX。
- **命令**：`scripts/chain_sprint7_distill.sh`（student 1500 迭代 + export_onnx）
- **⚠️ 导出后必须注入 ONNX 元数据**（10 条契约字段），否则板子启动报错。
  用 `released/bpx_pro_sprint3` 的 ONNX 当模板复制 metadata_props（见
  chain_full.sh 里的 inject 步骤）。
- **切换标准/验收**：
  - student play 复测极速/跟踪，和 teacher 一致（蒸馏保真）；
  - `scripts/verify_against_board.py --policy <student.onnx>` 逐位一致（≤1e-6）；
  - `scripts/check_contract.py --task ...Sprint7-Play-v0` 4/4 通过、观测 45 维。

---

## 4. 无辨识对照组怎么做

**走完全相同的四步，只把参数 yaml 从辨识换成无辨识。** 这是合法严谨的对照——两组
唯一差异是参数，其它步骤逐一对齐，差异可干净归因到"辨识 vs 无辨识"。

```bash
# 辨识组（默认用 data/bpx_pro_training_params.yaml）
BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml  bash scripts/chain_full.sh identified
# 无辨识对照（厂家宽先验，data/bpx_pro_baseline_params.yaml）
BPX_PRO_PARAMS_YAML=data/bpx_pro_baseline_params.yaml  bash scripts/chain_full.sh baseline
```

- 两组的 cfg、迭代数、种子策略、蒸馏流程**完全一样**，`chain_full.sh` 用环境变量选 yaml。
- 对照时逐步比：每步 play 复测同一批指标（抬脚/横滚/跟踪/极速），列成表。
- 已有的 `sprint4_baseline`（当年训到 475 迭代被叫停）就是无辨识链的早期产物，可弃。

---

## 5. 怎么加第五步（可续框架）

整条链是"继承 + 续训"的固定套路，加新能力照搬即可：

1. **写 cfg**：在 `velocity_env_cfg.py` 新建 `BpxProFlatSprint8EnvCfg(BpxProFlatSprint7EnvCfg)`
   （继承上一步，保留全部），`__post_init__` 里**只加新奖励/改新参数**，别动继承来的。
   配套写 `_PLAY` 变体（num_envs=50、关 corruption/push、固定指令）。
2. **注册**：`__init__.py` 加 `Sprint8-v0` 和 `Sprint8-Play-v0` 两个 gym.register。
3. **续训**：从上一步 teacher 种子续训（`--resume --load-run ".*上一步种子"`）。
   跨观测维度（如相位进观测 45→47）要先用 `scripts/migrate_phase_obs.py` 迁移权重。
4. **定切换标准**：想清楚"这步达标的量化指标是什么"（抬脚几 cm、横滚几度、极速多少），
   写进本文档对应小节。训完 play 复测，达标才固化。
5. **记录弯路**：失败的尝试（像二次型抬脚、sprint8 旋钮开到头）写进 cfg 注释和本文，
   当负结果留档，省得后人重踩。

**切换标准的统一原则**：这一步的新目标达标了、**且前面挣到的东西一个没丢**。具体就是
每步训完都 play 复测那几个核心指标，全过才进下一步。目前这个判断是人工看指标做的，
不是自动化 if-then——这是诚实的现状。

### ⚠️ 在 sprint7 上续训（加第五步）的两个坑

1. **种子用 teacher，不是 student**。续训（加新能力）必须从 **teacher** 续，不能用
   student。sprint7 teacher 是 `model_8255`：
   - 备份在 `released/bpx_pro_sprint7_teacher/model_8255.pt`（已锁死，优先用这个）；
   - 原始在 `logs/rsl_rl/bpx_pro_sprint7/2026-10-22_15-24-01_teacher/`。
   - ❌ 别用 `released/bpx_pro_sprint7/model_1499.pt`——那是 **student**（蒸馏产物），
     只能上板、不能续训（它的网络是冻结 base + 适应模块，续训会错）。
   续训命令形如：把 teacher 拷进新实验目录当种子，`--resume --load-run ".*种子"`。

2. **新步骤从 sprint9 起名，跳过 sprint8**。`BpxProFlatSprint8EnvCfg` 已被"失败版"占用
   （能量门控旋钮开到头 gate_min 0.03/v_hi 5.5/指令 6.0，极速没破 4.1、6.0 指令反而
   退化，确认撞了 33N·m 电机的物理天花板——训练产物已删，cfg 和注册留作负结果记录）。
   加新能力时**新建 `BpxProFlatSprint9EnvCfg(BpxProFlatSprint7EnvCfg)`**，别在 sprint8
   上接着改，免得和"撞墙"的旧配置混淆。

---

## 6. 关键文件索引

| 用途 | 文件 |
|---|---|
| 全链脚本（一键四步） | `scripts/chain_full.sh` |
| 蒸馏+导出 | `scripts/chain_sprint7_distill.sh` |
| 跨维度权重迁移 | `scripts/migrate_phase_obs.py`（相位进观测那类用） |
| 板端逐位对拍 | `scripts/verify_against_board.py` |
| 契约校验 | `scripts/check_contract.py` |
| play 复测（带 --report / --ramp / --side） | `scripts/play.py` |
| 环境 cfg（四步 + PLAY） | `bpx_rma/velocity_env_cfg.py` |
| 门控/抬脚/横滚奖励 | `bpx_rma/mdp/rewards.py` |
| 辨识参数 / 无辨识参数 | `data/bpx_pro_training_params.yaml` / `data/bpx_pro_baseline_params.yaml` |

## 7. 已验证的产物（保底，别删）

| 版本 | 位置 | 特点 |
|---|---|---|
| rollfix | `released/bpx_pro_sprint4_rollfix/` | 抬脚+横滚，极速 3.8 |
| **sprint7** | `released/bpx_pro_sprint7/` + `deploy_pkg/bpx_pro_sprint7/` | **当前最佳：极速 4.5，待上板** |
