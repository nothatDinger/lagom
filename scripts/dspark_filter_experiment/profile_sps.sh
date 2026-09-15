#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_env MODEL_PATH

mkdir -p "${RESULTS_ROOT}/sps" "${RESULTS_ROOT}/logs"
trap stop_server EXIT INT TERM

export SGLANG_DSPARK_ENABLE_SPS_RECORD=1
# SPS measures serving cost rather than model acceptance. Advancing every request
# by exactly the bonus token keeps the KV-length distribution deterministic; the
# profiler validates this value through /server_info before starting a sweep.
export SGLANG_SIMULATE_ACC_LEN=1.0
start_server static "${RESULTS_ROOT}/logs/sps-server.log"

# shellcheck disable=SC2206
extra_sps_args=(${EXTRA_SPS_ARGS})
python -m sglang.benchmark.dspark_sps_profiler run \
  --base-url "http://${HOST}:${PORT}" \
  --max-batch-size "${SPS_MAX_BATCH_SIZE}" \
  --repeats "${SPS_REPEATS}" \
  --out "${SPS_TABLE_PATH}" \
  "${extra_sps_args[@]}" 2>&1 | tee "${RESULTS_ROOT}/logs/sps-profiler.log"

echo "SPS table: ${SPS_TABLE_PATH}"
