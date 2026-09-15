#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/dspark_filter_experiment}"
SPS_TABLE_PATH="${SPS_TABLE_PATH:-${RESULTS_ROOT}/sps/dspark-sps.json}"
SERVER_PID=""
SERVER_LOG_PATH=""

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "error: export ${name} before running this script" >&2
      exit 2
    fi
  done
}

wait_for_server() {
  local interval=5
  local tries=$(((SERVER_READY_TIMEOUT_SECONDS + interval - 1) / interval))
  for ((i = 1; i <= tries; i++)); do
    if curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "error: server exited before becoming healthy" >&2
      print_server_log_tail
      return 1
    fi
    sleep "${interval}"
  done
  echo "error: server was not healthy after ${SERVER_READY_TIMEOUT_SECONDS} seconds" >&2
  print_server_log_tail
  return 1
}

print_server_log_tail() {
  if [[ -n "${SERVER_LOG_PATH}" && -f "${SERVER_LOG_PATH}" ]]; then
    echo "--- last 200 lines of ${SERVER_LOG_PATH} ---" >&2
    tail -n 200 "${SERVER_LOG_PATH}" >&2
    echo "--- end server log ---" >&2
  fi
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    # Give SGLang's multiprocessing children time to unlink semaphores/shared
    # memory. SIGKILL is a last resort and may still produce resource_tracker
    # warnings, but it prevents a wedged server from retaining the GPUQ job.
    kill -TERM "${SERVER_PID}"
    local deadline=$((SECONDS + SERVER_STOP_TIMEOUT_SECONDS))
    while kill -0 "${SERVER_PID}" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "warning: server did not stop gracefully; sending SIGKILL" >&2
      kill -KILL "${SERVER_PID}"
    fi
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

start_server() {
  local mode="$1"
  local log_path="$2"
  SERVER_LOG_PATH="${log_path}"
  shift 2
  # EXTRA_SERVER_ARGS is explicitly an operator-provided shell fragment.
  # shellcheck disable=SC2206
  local extra_server_args=(${EXTRA_SERVER_ARGS})
  PYTHONUNBUFFERED=1 SGLANG_RAGGED_VERIFY_MODE="${mode}" \
    sglang serve \
      --trust-remote-code \
      --model-path "${MODEL_PATH}" \
      --tp "${TP_SIZE}" \
      --moe-runner-backend "${MOE_RUNNER_BACKEND}" \
      --speculative-algorithm DSPARK \
      --host "${HOST}" \
      --port "${PORT}" \
      "${extra_server_args[@]}" \
      "$@" >"${log_path}" 2>&1 &
  SERVER_PID=$!
  wait_for_server
}
