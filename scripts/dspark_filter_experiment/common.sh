#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=config.sh
source "${SCRIPT_DIR}/config.sh"

RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/dspark_filter_experiment}"
SPS_TABLE_PATH="${SPS_TABLE_PATH:-${RESULTS_ROOT}/sps/dspark-sps.json}"
SERVER_PID=""

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
  local tries="${SERVER_READY_TRIES:-180}"
  for ((i = 1; i <= tries; i++)); do
    if curl --fail --silent "http://${HOST}:${PORT}/health" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "error: server exited before becoming healthy" >&2
      return 1
    fi
    sleep 5
  done
  echo "error: server was not healthy after $((tries * 5)) seconds" >&2
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}"
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

start_server() {
  local mode="$1"
  local log_path="$2"
  shift 2
  # EXTRA_SERVER_ARGS is explicitly an operator-provided shell fragment.
  # shellcheck disable=SC2206
  local extra_server_args=(${EXTRA_SERVER_ARGS})
  SGLANG_RAGGED_VERIFY_MODE="${mode}" \
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
