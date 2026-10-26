#!/bin/bash
# Sprint9 teacher 续训：种子 = sprint7 teacher model_8255。
# 改动理由见 velocity_env_cfg.BpxProFlatSprint9EnvCfg 的 docstring；框架见 TRAINING.md §5。
#
#   bash scripts/chain_sprint9.sh            # 辨识组，1200 迭代
#   bash scripts/chain_sprint9.sh 2000       # 自定迭代数
#   BPX_PRO_PARAMS_YAML=data/bpx_pro_baseline_params.yaml bash scripts/chain_sprint9.sh   # 无辨识对照
#
# ⚠️ 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION 会静默打死脚本。

cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[s9] conda activate 失败"; exit 1; }

EXP=bpx_pro_sprint9
YAML="${BPX_PRO_PARAMS_YAML:-data/bpx_pro_training_params.yaml}"
ITERS="${1:-1200}"
SEED_SRC=released/bpx_pro_sprint7_teacher
SEED_DST="logs/rsl_rl/${EXP}/s7seed_teacher"

export BPX_PRO_PARAMS_YAML="${YAML}"
echo "[s9] $(date '+%F %T') 参数 ${YAML} / ${ITERS} 迭代"

# 种子必须是 teacher。❌ released/bpx_pro_sprint7/model_1499.pt 是 student（冻结 base
# + 适应模块），续训会错。见 TRAINING.md §5 的两个坑。
if [ ! -d "${SEED_DST}" ]; then
    [ -f "${SEED_SRC}/model_8255.pt" ] || { echo "[s9] 找不到 ${SEED_SRC}/model_8255.pt"; exit 1; }
    mkdir -p "logs/rsl_rl/${EXP}"
    cp -r "${SEED_SRC}" "${SEED_DST}" || exit 1
    echo "[s9] 种子就位：${SEED_SRC} -> ${SEED_DST}"
else
    echo "[s9] 种子已存在，复用 ${SEED_DST}"
fi

python -u scripts/train.py --headless --task Isaac-Velocity-Flat-BpxPro-Sprint9-v0 \
    --num_envs 4096 --max_iterations "${ITERS}" \
    --resume --load-run ".*s7seed_teacher" \
    --experiment-name "${EXP}" --run-name teacher \
    > logs/train_teacher_sprint9.log 2>&1
rc=$?; echo "[s9] $(date '+%F %T') teacher 退出码 ${rc}"
[ ${rc} -ne 0 ] && { echo "[s9] 中止，看 logs/train_teacher_sprint9.log"; exit 1; }

cat <<'EOT'

[s9] 训完先复测，三项全过才蒸馏（teacher 没有 ONNX，所以用 play.py 不是 sim2sim）：

  P=Isaac-Velocity-Flat-BpxPro-Sprint9-Play-v0
  A=rsl_rl_cfg_entry_point

  # ① 冷启动（这一版的主要目标）。--ramp 默认 0 = 阶跃指令，就是手柄一把推到底。
  #    sprint7 的基线：4.0 能起步、4.5 以上全站桩（实速 0.00）。目标是 6.0 也能起步。
  python scripts/play.py --headless --task $P --agent $A --vx 6.0 --report
  python scripts/play.py --headless --task $P --agent $A --vx 4.5 --report

  # ② 极速（斜坡）。sprint7 在新包络下是 5.37，目标 6.0。
  python scripts/play.py --headless --task $P --agent $A --vx 6.5 --ramp 0.7 --report

  # ③ 别丢前面挣到的：抬脚 foot_apex ≥6cm、横滚 <1°、低速跟踪 <0.12
  python scripts/play.py --headless --task $P --agent $A --vx 1.5 --report

[s9] 三项达标后蒸馏 + 导出（照 chain_sprint7_distill.sh 的套路，把 task 换成 Sprint9）：
     注意导出后要注入 ONNX 元数据，否则板子启动报错（TRAINING.md §3④）。
     student 出来后才能用 sim2sim.py 的 coldstart / topspeed6 预设跨引擎复核。
EOT
