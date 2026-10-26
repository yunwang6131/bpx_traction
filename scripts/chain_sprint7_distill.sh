#!/bin/bash
# Sprint7 蒸馏 + 导出（teacher = rollfix + 能量门控，极速 ~4.5，45 维不碰板子）。
cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[chain] conda 失败"; exit 1; }
EXP=bpx_pro_sprint7; YAML=data/bpx_pro_training_params.yaml
TASK=Isaac-Velocity-Flat-BpxPro-Sprint7-v0

echo "[chain] $(date '+%F %T') ${EXP} / student (1500 迭代，teacher=15-24-01)"
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/train.py --headless \
    --task "${TASK}" --num_envs 4096 --max_iterations 1500 \
    --agent rsl_rl_distillation_cfg_entry_point --load-run ".*15-24-01_teacher" \
    --experiment-name "${EXP}" --run-name student > logs/train_student_sprint7.log 2>&1
rc=$?; echo "[chain] $(date '+%F %T') student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
BPX_PRO_PARAMS_YAML="${YAML}" python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint7-Play-v0 --experiment-name "${EXP}" \
    > logs/chain_s7_export.log 2>&1
echo "[chain] $(date '+%F %T') 导出退出码 $?  蒸馏链完成"
