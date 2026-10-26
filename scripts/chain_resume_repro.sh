#!/bin/bash
# 复现验证——接力版：复用今天已跑完的 sprint3 从零种子，续 ②抬脚+横滚 → ③门控 → ④蒸馏。
# 用独立实验名 bpx_pro_repro_* ，不碰手动跑的 sprint7。
# 每个训练步带自动重试（兜住 Isaac 启动时偶发的 USD 下载网络崩）。
#
#   cd /home/wangyun/intern/ZGYD/bpx
#   nohup systemd-inhibit --what=sleep --why="bpx repro接力" bash scripts/chain_resume_repro.sh > logs/chain_repro.log 2>&1 &
#
# 不要加 set -u：Isaac 的 setup_conda_env.sh 无保护引用 $ZSH_VERSION 会静默打死脚本。

cd /home/wangyun/intern/ZGYD/bpx || exit 1
source ~/anaconda3/etc/profile.d/conda.sh
conda activate bpx_rma || { echo "[repro] conda 失败"; exit 1; }
export BPX_PRO_PARAMS_YAML=data/bpx_pro_training_params.yaml   # 辨识参数

# 今天已跑完的 sprint3 从零种子（接力起点）
S3SEED="logs/rsl_rl/bpx_pro_sprint3/2026-10-23_09-44-40_teacher"
[ -f "$S3SEED/model_4199.pt" ] || { echo "[repro] 找不到 sprint3 种子 $S3SEED"; exit 1; }

E4=bpx_pro_repro_sprint4
E7=bpx_pro_repro_sprint7

# 带重试的训练：崩了(含 USD 下载网络崩)最多重试 3 次
train_retry() {  # train_retry <实验名> <task> <迭代> <run> <种子正则> <日志>
    local exp=$1 task=$2 iters=$3 run=$4 seed=$5 log=$6
    for try in 1 2 3; do
        echo "[repro] $(date '+%F %T') 训练 ${exp}/${run} (第${try}次尝试, ${iters}迭代)"
        python -u scripts/train.py --headless --task "${task}" --num_envs 4096 \
            --max_iterations "${iters}" --resume --load-run "${seed}" \
            --experiment-name "${exp}" --run-name "${run}" > "${log}" 2>&1
        local rc=$?
        # 成功判据：退出码0 且 日志里有接近目标的迭代数（真训了，不是秒退）
        if [ ${rc} -eq 0 ] && grep -q "iteration $((iters-1))/" "${log}"; then
            echo "[repro] $(date '+%F %T') ${exp}/${run} ✓ 成功"; sleep 20; return 0
        fi
        echo "[repro] ${exp}/${run} 第${try}次失败(rc=${rc})，20s 后重试"
        grep -m1 -iE "FileNotFoundError|usd file|Traceback" "${log}" | head -1
        sleep 20
    done
    echo "[repro] ${exp}/${run} 重试3次仍失败，中止"; exit 1
}

seed_into() {  # 把某目录拷成新实验的种子
    local src=$1 dst_exp=$2 dst_run=$3
    mkdir -p "logs/rsl_rl/${dst_exp}"
    rm -rf "logs/rsl_rl/${dst_exp}/${dst_run}"
    cp -r "$src" "logs/rsl_rl/${dst_exp}/${dst_run}"
}

# ② sprint4 抬脚+横滚（种子=今天的 sprint3）
seed_into "$S3SEED" "$E4" "s3seed_teacher"
train_retry "$E4" "Isaac-Velocity-Flat-BpxPro-Sprint4-v0" 1500 "teacher" ".*s3seed_teacher" "logs/repro_sprint4.log"

# ③ sprint7 门控（种子=上一步 sprint4）
S4=$(ls -dt logs/rsl_rl/${E4}/*teacher | grep -v s3seed | head -1)
seed_into "$S4" "$E7" "s4seed_teacher"
train_retry "$E7" "Isaac-Velocity-Flat-BpxPro-Sprint7-v0" 1200 "teacher" ".*s4seed_teacher" "logs/repro_sprint7.log"

# ④ 蒸馏 + 导出（带重试）
for try in 1 2 3; do
    echo "[repro] $(date '+%F %T') 蒸馏 ${E7}/student (第${try}次)"
    python -u scripts/train.py --headless --task Isaac-Velocity-Flat-BpxPro-Sprint7-v0 \
        --num_envs 4096 --max_iterations 1500 --agent rsl_rl_distillation_cfg_entry_point \
        --load-run ".*_teacher" --experiment-name "${E7}" --run-name student \
        > logs/repro_student.log 2>&1
    [ $? -eq 0 ] && grep -q "iteration 1499/" logs/repro_student.log && { echo "[repro] 蒸馏 ✓"; break; }
    echo "[repro] 蒸馏第${try}次失败，重试"; sleep 20
done
sleep 20
python -u scripts/export_onnx.py --headless \
    --task Isaac-Velocity-Flat-BpxPro-Sprint7-Play-v0 --experiment-name "${E7}" \
    > logs/repro_export.log 2>&1
# 元数据注入
STU=$(ls -d logs/rsl_rl/${E7}/*student | head -1)
python - "$STU/exported/policy.onnx" <<'PY'
import sys, onnx
src = onnx.load("deploy_pkg/bpx_pro_sprint3/exported/policy.onnx")
dst = onnx.load(sys.argv[1]); del dst.metadata_props[:]
for p in src.metadata_props:
    q = dst.metadata_props.add(); q.key, q.value = p.key, p.value
onnx.save(dst, sys.argv[1]); print(f"[repro] 注入 {len(dst.metadata_props)} 条元数据")
PY

echo "[repro] $(date '+%F %T') ✅ 复现接力完成。ONNX: ${STU}/exported/policy.onnx"
