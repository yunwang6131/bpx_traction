#!/bin/bash
# Sprint9 蒸馏 + 导出：teacher -> student（45 维可上板）-> ONNX -> 注入元数据。
#
# 前置：chain_sprint9.sh 的 teacher 已训完**并且 play 复测三项达标**（冷启动 6.0 能起步 /
# 斜坡极速 ≥6.0 / 低速抬脚横滚跟踪没退化，见 TRAINING.md §3）。不达标别蒸，白烧 1500 迭代。
#
#   bash scripts/chain_sprint9_distill.sh          # 1500 迭代
#   bash scripts/chain_sprint9_distill.sh 2000
#
# ⚠️ --load-run 用的是**精确 run 名**，不是 ".*_teacher"：isaaclab 的 get_checkpoint_path
#    默认 sort_alpha=True，按字母序取最后一个匹配项，而种子目录 "s7seed_teacher" 的字母序
#    排在时间戳目录 "2026-..." 之后（数字 ASCII < 字母）—— 用通配会选中种子，蒸出来的
#    student 就对错了 teacher。chain_full.sh 第④步有同样的问题。
# ⚠️ 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION 会静默打死脚本。

cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[s9d] conda activate 失败"; exit 1; }

EXP=bpx_pro_sprint9
YAML="${BPX_PRO_PARAMS_YAML:-data/bpx_pro_training_params.yaml}"
TASK=Isaac-Velocity-Flat-BpxPro-Sprint9-v0
PLAY_TASK=Isaac-Velocity-Flat-BpxPro-Sprint9-Play-v0
ITERS="${1:-1500}"
export BPX_PRO_PARAMS_YAML="${YAML}"

# 排除种子目录，按修改时间取最新的真实 teacher run
TEACHER=$(ls -dt logs/rsl_rl/${EXP}/*_teacher 2>/dev/null | grep -v 's7seed_teacher' | head -1)
[ -z "${TEACHER}" ] && { echo "[s9d] 找不到 ${EXP} 的 teacher run —— 先跑 chain_sprint9.sh"; exit 1; }
TRUN=$(basename "${TEACHER}")
echo "[s9d] $(date '+%F %T') teacher=${TRUN}  student ${ITERS} 迭代  参数 ${YAML}"

python -u scripts/train.py --headless --task "${TASK}" \
    --num_envs 4096 --max_iterations "${ITERS}" \
    --agent rsl_rl_distillation_cfg_entry_point \
    --load-run "^${TRUN}$" \
    --experiment-name "${EXP}" --run-name student \
    > logs/train_student_sprint9.log 2>&1
rc=$?; echo "[s9d] $(date '+%F %T') student 退出码 ${rc}"
[ ${rc} -ne 0 ] && { echo "[s9d] 中止，看 logs/train_student_sprint9.log"; exit 1; }
sleep 20

python -u scripts/export_onnx.py --headless \
    --task "${PLAY_TASK}" --experiment-name "${EXP}" \
    > logs/chain_s9_export.log 2>&1
echo "[s9d] 导出退出码 $?"

# ⚠️ 必须注入 ONNX 元数据（10 条契约字段），否则板子启动报错。见 TRAINING.md §3④
STU=$(ls -dt logs/rsl_rl/${EXP}/*student 2>/dev/null | head -1)
[ -z "${STU}" ] && { echo "[s9d] 找不到 student run，元数据没注入"; exit 1; }
python - "${STU}/exported/policy.onnx" <<'PY'
import sys, onnx
src = onnx.load("deploy_pkg/bpx_pro_sprint3/exported/policy.onnx")
dst = onnx.load(sys.argv[1]); del dst.metadata_props[:]
for p in src.metadata_props:
    q = dst.metadata_props.add(); q.key, q.value = p.key, p.value
onnx.save(dst, sys.argv[1]); print(f"[s9d] 注入 {len(dst.metadata_props)} 条元数据")
PY

cat <<EOT

[s9d] student 出来了：${STU}/exported/policy.onnx
[s9d] 上板前三道校验：
  python scripts/check_contract.py --headless --task ${PLAY_TASK}
  python scripts/verify_against_board.py --policy ${STU}/exported/policy.onnx

[s9d] 跨引擎复核（这两个预设就是 sprint9 的验收基线）：
  # 冷启动：sprint7 基线是 4.0 能起步、4.5 以上实速 0.00。sprint9 目标是 6.0 也能起步。
  python scripts/sim2sim.py --policy s9=${STU}/exported/policy.onnx \\
      --preset coldstart --seg-warmup 2.0 --out logs/sim2sim_s9_cold
  # 极速：sprint7 在新包络下是 5.37，目标 6.0。
  python scripts/sim2sim.py --policy s9=${STU}/exported/policy.onnx \\
      --preset topspeed6 --seg-warmup 2.0 --out logs/sim2sim_s9_top
EOT
