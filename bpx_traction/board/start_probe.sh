#!/bin/bash
# 后台启动传感器探测，扛过"网线从开发机挪到机器人"那次断链。
#
# 和 v4 的 start.sh 同一个套路：在开发机还连着板子的时候启动，进程 detach，
# SSH 断掉不影响它；它一直重试连接，等机器人上线之后开始探，结果写日志文件。
#
#   ./start_probe.sh            # 在板子上执行（此时板子还连着开发机）
#   # 然后拔网线插机器人，等 ~30 秒
#   ./stop_probe.sh             # 或者等它自己跑完（默认 10 s 采集 + 退出）
#
# 纯只读：用 RequestRobotState 被动订阅状态，一条控制指令都不发。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p log
PID_FILE=log/probe_sensors.pid
LOG_FILE=log/probe_sensors.log
OUT_FILE=log/probe_sensors.result

pid_matches() {
  local pid=${1:-}
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- 'probe_sensors.py'
}

if [ -f "$PID_FILE" ]; then
  old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
  if pid_matches "$old_pid"; then
    echo "已在运行：pid=$old_pid"
    echo "日志：tail -f $LOG_FILE"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

# 探测只读，但两个 SDK 客户端同时连同一台机器人仍然可能互相干扰。
if ps -C python3 -o args= 2>/dev/null | grep -Fq -- 'bpx_dual_pad.py'; then
  echo "错误：bpx_dual_pad.py 正在运行。先 ./stop.sh 停掉它，" >&2
  echo "      避免两个 SDK 客户端争用同一台机器人。" >&2
  exit 1
fi

python3 -m py_compile probe_sensors.py

{
  echo
  echo "===== probe $(date -Iseconds) ====="
} >> "$LOG_FILE"
nohup python3 -u ./probe_sensors.py --out "$OUT_FILE" "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 1
if ! pid_matches "$pid"; then
  echo "错误：探测进程启动后立即退出。" >&2
  tail -40 "$LOG_FILE" >&2 || true
  rm -f "$PID_FILE"
  exit 1
fi

cat <<EOF
已在后台等待机器人上线，pid=$pid。

现在可以拔掉开发机网线，把 eth0 接到机器人。SSH 断开不影响这个进程。

连上之后会采 10 秒，结果写在：
    $OUT_FILE      （结论，就是这份要发回去）
    $LOG_FILE      （完整输出）

跑完把网线插回开发机，然后在开发机上：
    scp 10.21.20.100:~/wy/<这个目录>/$OUT_FILE .
EOF
