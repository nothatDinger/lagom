#!/usr/bin/env bash
# Monitor one experiment from the same GPU node; use --watch to repeat.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
run_dir=${1:-"$ROOT/results/deepseek_v4_csa_topk_h2d_batchsize/latest"}
watch_mode=${2:-}
STALL_THRESHOLD_SEC=${STALL_THRESHOLD_SEC:-1200}
GPU_BUSY_THRESHOLD=${GPU_BUSY_THRESHOLD:-10}

read_state() { sed -n "s/^$1=//p" "$run_dir/run_state.env" 2>/dev/null | tail -1; }

check_once() {
  local now state batch_size mode pid newest newest_mtime age health gpu_busy
  now=$(date +%s)
  state=$(read_state state); batch_size=$(read_state batch_size); mode=$(read_state mode); pid=$(read_state server_pid)
  printf 'run=%s state=%s batch_size=%s mode=%s pid=%s\n' "$run_dir" "${state:-unknown}" "${batch_size:--}" "${mode:--}" "${pid:--}"

  if [[ -n "$pid" && "$pid" != "-" ]] && ! kill -0 "$pid" 2>/dev/null; then
    echo "ERROR: recorded server process is not alive"
    return 3
  fi

  health=down
  if curl -fsS --max-time 2 "http://${HOST:-127.0.0.1}:${PORT:-30000}/health" >/dev/null 2>&1; then health=ready; fi
  echo "health=$health"

  newest=$(find -L "$run_dir" -type f \( -name 'server_*.log' -o -name 'server_*.err' -o -name 'client_*.log' -o -name 'h2d_trace.tp0.jsonl' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
  if [[ -n "$newest" ]]; then
    newest_mtime=$(stat -c %Y "$newest"); age=$((now - newest_mtime))
    printf 'latest_file=%s age_sec=%s size_bytes=%s\n' "$newest" "$age" "$(stat -c %s "$newest")"
    tail -n 5 "$newest" || true
  else
    age=$((now - $(stat -c %Y "$run_dir")))
    printf 'latest_file=none age_sec=%s\n' "$age"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu_index, utilization_gpu_pct, memory_used_MiB, memory_total_MiB"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    gpu_busy=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk -v t="$GPU_BUSY_THRESHOLD" '$1 >= t {busy=1} END {print busy+0}')
    if [[ "$health" == down && "$age" -ge "$STALL_THRESHOLD_SEC" && "$gpu_busy" -eq 0 ]]; then
      echo "STUCK-SUSPECTED: no health response or file progress for ${age}s and every GPU is below ${GPU_BUSY_THRESHOLD}% utilization"
      return 2
    fi
  elif [[ "$health" == down && "$age" -ge "$STALL_THRESHOLD_SEC" ]]; then
    echo "WARNING: no progress for ${age}s, but nvidia-smi is unavailable; cannot distinguish compilation from a hang"
  fi
  echo "status=progressing-or-ready"
}

if [[ "$watch_mode" == "--watch" ]]; then
  while true; do date -u; check_once || true; echo; sleep "${WATCH_INTERVAL_SEC:-30}"; done
else
  check_once
fi
