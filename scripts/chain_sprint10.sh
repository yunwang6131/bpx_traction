#!/bin/bash
# Sprint10 teacher 续训：种子 = sprint9 teacher（冷启动阈值 5.0 那版）。
# 改动理由见 velocity_env_cfg.BpxProFlatSprint10EnvCfg 的 docstring。
#
#   bash scripts/chain_sprint10.sh              # 1500 迭代
#   bash scripts/chain_sprint10.sh 2000
#   BPX_S10_EXP=bpx_pro_sprint10b bash scripts/chain_sprint10.sh    # 换旋钮时隔离日志
#
# ⚠️ 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION 会静默打死脚本。

cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[s10] conda activate 失败"; exit 1; }

EXP="${BPX_S10_EXP:-bpx_pro_sprint10}"
YAML="${BPX_PRO_PARAMS_YAML:-data/bpx_pro_training_params.yaml}"
ITERS="${1:-1500}"
SEED_DST="logs/rsl_rl/${EXP}/s9seed_teacher"

export BPX_PRO_PARAMS_YAML="${YAML}"
echo "[s10] $(date '+%F %T') 实验 ${EXP} / ${ITERS} 迭代 / 参数 ${YAML}"

# 种子 = sprint9 的 teacher（不是 student）。排除它自己的种子目录，按时间取最新。
if [ ! -d "${SEED_DST}" ]; then
    SRC=$(ls -dt logs/rsl_rl/bpx_pro_sprint9/*_teacher 2>/dev/null | grep -v seed | head -1)
    [ -z "${SRC}" ] && { echo "[s10] 找不到 sprint9 的 teacher run"; exit 1; }
    mkdir -p "logs/rsl_rl/${EXP}"
    cp -r "${SRC}" "${SEED_DST}" || exit 1
    echo "[s10] 种子就位：${SRC} -> ${SEED_DST}"
else
    echo "[s10] 种子已存在，复用 ${SEED_DST}"
fi

python -u scripts/train.py --headless --task Isaac-Velocity-Flat-BpxPro-Sprint10-v0 \
    --num_envs 4096 --max_iterations "${ITERS}" \
    --resume --load-run ".*s9seed_teacher" \
    --experiment-name "${EXP}" --run-name teacher \
    > "logs/train_teacher_${EXP#bpx_pro_}.log" 2>&1
rc=$?; echo "[s10] $(date '+%F %T') teacher 退出码 ${rc}"
[ ${rc} -ne 0 ] && { echo "[s10] 中止，看 logs/train_teacher_${EXP#bpx_pro_}.log"; exit 1; }

cat <<'EOT'

[s10] 四向验收（teacher 没有 ONNX，用 play.py；标准见 cfg docstring）：

  P=Isaac-Velocity-Flat-BpxPro-Sprint10-Play-v0
  A=rsl_rl_cfg_entry_point

  # 前进：冷启动阈值保持 ≥5.0，且不能再出现 sprint9 那种高速驻留翻车
  python scripts/play.py --headless --task $P --agent $A --vx 5.0 --report
  # 后退：目标实速 ≥2.5、抬脚 ≥6 cm（sprint9 基线 −1.48 / 4.6 cm）
  python scripts/play.py --headless --task $P --agent $A --vx -3.0 --report
  # 横移：1.5 要抬脚 ≥6cm 横滚 <2°；0.2 要步频 >1 Hz（sprint9 基线 0.25 Hz）
  python scripts/play.py --headless --task $P --agent $A --vx 0 --vy 1.5 --report
  python scripts/play.py --headless --task $P --agent $A --vx 0 --vy 0.2 --report
  # 转向：2.0 抬脚 ≥6cm；3.0 误差 <0.2（sprint9 基线 4.5 cm / 0.145）
  python scripts/play.py --headless --task $P --agent $A --vx 0 --wz 3.0 --report

[s10] 蒸馏前务必跑高速长驻留安全验证（6 秒采样不足以判安全，sprint9 就是栽在这）：
  python scripts/sim2sim.py --policy s10=<student onnx> --preset dwell --seg-warmup 2.0 --out logs/sim2sim_s10_dwell
EOT
