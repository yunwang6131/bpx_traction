#!/bin/bash
# 后退专项判别实验（**不是产品版本，不要蒸馏上板**）。
# 目的见 velocity_env_cfg.BpxProFlatSprint10BackEnvCfg 的 docstring：把"物理做不到"
# 和"多任务里后退性价比不够"分开。种子 = sprint10 的 teacher，只续 400 迭代。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[s10b] conda 失败"; exit 1; }

EXP=bpx_pro_sprint10_back
ITERS="${1:-400}"
SEED_DST="logs/rsl_rl/${EXP}/s10seed_teacher"
export BPX_PRO_PARAMS_YAML="${BPX_PRO_PARAMS_YAML:-data/bpx_pro_training_params.yaml}"

if [ ! -d "${SEED_DST}" ]; then
    SRC=$(ls -dt logs/rsl_rl/bpx_pro_sprint10/*_teacher 2>/dev/null | grep -v seed | head -1)
    [ -z "${SRC}" ] && { echo "[s10b] 找不到 sprint10 teacher"; exit 1; }
    mkdir -p "logs/rsl_rl/${EXP}"; cp -r "${SRC}" "${SEED_DST}" || exit 1
    echo "[s10b] 种子：${SRC}"
fi

echo "[s10b] $(date '+%F %T') 后退专项 ${ITERS} 迭代"
python -u scripts/train.py --headless --task Isaac-Velocity-Flat-BpxPro-Sprint10Back-v0 \
    --num_envs 4096 --max_iterations "${ITERS}" \
    --resume --load-run ".*s10seed_teacher" \
    --experiment-name "${EXP}" --run-name teacher \
    > logs/train_teacher_sprint10_back.log 2>&1
rc=$?; echo "[s10b] $(date '+%F %T') 退出码 ${rc}"
