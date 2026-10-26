#!/bin/bash
# BPX-Pro 全链训练（四步课程 + 蒸馏 + 导出），一键从零跑到可上板的 ONNX。
# 配方详解见 TRAINING.md。这是"模式 A：从零一键跑完整链"。
#
#   辨识组:    bash scripts/chain_full.sh identified
#   无辨识对照: bash scripts/chain_full.sh baseline
#
# 唯一区别是参数 yaml（辨识 vs 厂家宽先验）；cfg、迭代数、流程完全一样。
#
# ⚠️ 风险（见 TRAINING.md §2）：① sprint3 从零有起飞随机性；② Sprint4 一次学抬脚+横滚
#    未单独验证。某步 play 复测不达标就退回模式 B（从种子 checkpoint 分步接力）。
# ⚠️ 不要加 set -u：Isaac Sim 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION 会静默打死脚本。

cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[full] conda activate 失败"; exit 1; }

# ---- 参数组选择 ----
MODE="${1:-identified}"
case "$MODE" in
  identified) YAML=data/bpx_pro_training_params.yaml; SUF="" ;;
  baseline)   YAML=data/bpx_pro_baseline_params.yaml;  SUF="_baseline" ;;
  *) echo "用法: $0 [identified|baseline]"; exit 1 ;;
esac
export BPX_PRO_PARAMS_YAML="${YAML}"
echo "[full] 模式=${MODE}  参数=${YAML}"

# 四个实验名（无辨识组带 _baseline 后缀，和辨识组隔离）
E3="bpx_pro_sprint3${SUF}"
E4="bpx_pro_sprint4${SUF}"
E7="bpx_pro_sprint7${SUF}"

train() {  # train <实验名> <task> <迭代> <run名> [种子正则]
    local exp=$1 task=$2 iters=$3 run=$4 seed=$5
    local extra=""
    [ -n "$seed" ] && extra="--resume --load-run ${seed}"
    echo "[full] $(date '+%F %T') 训练 ${exp}/${run} (${iters} 迭代) ${seed:+续训自 $seed}"
    python -u scripts/train.py --headless --task "${task}" --num_envs 4096 \
        --max_iterations "${iters}" ${extra} \
        --experiment-name "${exp}" --run-name "${run}" > "logs/full_${exp}_${run}.log" 2>&1
    local rc=$?; echo "[full] $(date '+%F %T') ${exp}/${run} 退出码 ${rc}"
    [ ${rc} -ne 0 ] && { echo "[full] 中止"; exit 1; }
    sleep 20
}

seed_from() {  # 把某实验最新 teacher 的最终 model 软链成下一步的种子目录
    local src_exp=$1 dst_exp=$2 dst_run=$3
    local src=$(ls -dt logs/rsl_rl/${src_exp}/*teacher 2>/dev/null | head -1)
    [ -z "$src" ] && { echo "[full] 找不到 ${src_exp} 的 teacher"; exit 1; }
    mkdir -p "logs/rsl_rl/${dst_exp}"
    cp -r "$src" "logs/rsl_rl/${dst_exp}/${dst_run}"
    echo "[full] 种子: ${src} -> ${dst_exp}/${dst_run}"
}

# ============ ① 基础: sprint3 从零 4200 ============
train "${E3}" "Isaac-Velocity-Flat-BpxPro-Sprint3-v0" 4200 "teacher"
# 复测提示: play --task Sprint3-Play-v0 --vx 1.5 --report  → 跟踪 <0.1、起飞稳住

# ============ ② 抬脚+横滚: Sprint4 续训（种子=sprint3）============
seed_from "${E3}" "${E4}" "step1seed_teacher"
train "${E4}" "Isaac-Velocity-Flat-BpxPro-Sprint4-v0" 1500 "teacher" ".*step1seed_teacher"
# 切换标准: foot_apex ≥6cm 且 横滚 <1° 且 跟踪 <0.12（见 TRAINING.md §3②）

# ============ ③ 能量门控: Sprint7 续训（种子=sprint4）============
seed_from "${E4}" "${E7}" "step2seed_teacher"
train "${E7}" "Isaac-Velocity-Flat-BpxPro-Sprint7-v0" 1200 "teacher" ".*step2seed_teacher"
# 切换标准: 斜坡极速 ≥4.3 且 低速不退化（见 TRAINING.md §3③）

# ============ ④ 蒸馏 + 导出 ============
echo "[full] $(date '+%F %T') 蒸馏 ${E7}/student (1500)"
python -u scripts/train.py --headless --task Isaac-Velocity-Flat-BpxPro-Sprint7-v0 \
    --num_envs 4096 --max_iterations 1500 --agent rsl_rl_distillation_cfg_entry_point \
    --load-run ".*_teacher" --experiment-name "${E7}" --run-name student \
    > "logs/full_${E7}_student.log" 2>&1
rc=$?; echo "[full] student 退出码 ${rc}"; [ ${rc} -ne 0 ] && exit 1
sleep 20
python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint7-Play-v0 --experiment-name "${E7}" \
    > "logs/full_${E7}_export.log" 2>&1
echo "[full] 导出退出码 $?"

# ⚠️ 导出后必须注入 ONNX 元数据（见 TRAINING.md §3④），否则板子启动报错
STU=$(ls -d logs/rsl_rl/${E7}/*student 2>/dev/null | head -1)
python - "$STU/exported/policy.onnx" <<'PY'
import sys, onnx
src = onnx.load("deploy_pkg/bpx_pro_sprint3/exported/policy.onnx")
dst = onnx.load(sys.argv[1]); del dst.metadata_props[:]
for p in src.metadata_props:
    q = dst.metadata_props.add(); q.key, q.value = p.key, p.value
onnx.save(dst, sys.argv[1]); print(f"[full] 注入 {len(dst.metadata_props)} 条元数据")
PY

echo "[full] $(date '+%F %T') ✅ 全链完成。ONNX: ${STU}/exported/policy.onnx"
echo "[full] 上板前跑: verify_against_board.py + check_contract.py 校验"
