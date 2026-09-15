#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_env MODEL_PATH DATASET_PATH

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "error: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
  exit 2
fi
if [[ ! -s "${SPS_TABLE_PATH}" ]]; then
  echo "error: SPS table not found: ${SPS_TABLE_PATH}; run profile_sps.sh first" >&2
  exit 2
fi

mkdir -p "${RESULTS_ROOT}/raw" "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/bench" "${RESULTS_ROOT}/reports"
trap stop_server EXIT INT TERM

dataset_for_run="${RESULTS_ROOT}/datasets/${DATASET_NAME}-${DATASET_SAMPLING}-${DATASET_SAMPLE_SIZE}.json"
python "${SCRIPT_DIR}/prepare_dataset.py" \
  --input "${DATASET_PATH}" \
  --output "${dataset_for_run}" \
  --strategy "${DATASET_SAMPLING}" \
  --size "${DATASET_SAMPLE_SIZE}" \
  --seed "${DATASET_SAMPLE_SEED}" \
  2>&1 | tee "${RESULTS_ROOT}/logs/dataset-preparation.log"

records=()
for ((run = 1; run <= RUN_COUNT; run++)); do
  record_path="${RESULTS_ROOT}/raw/run-${run}.jsonl"
  server_log="${RESULTS_ROOT}/logs/run-${run}-server.log"
  bench_log="${RESULTS_ROOT}/logs/run-${run}-benchmark.log"
  bench_result="${RESULTS_ROOT}/bench/run-${run}.jsonl"
  seed=$((BASE_SEED + run - 1))
  rm -f "${record_path}" "${bench_result}"

  export SGLANG_DSPARK_FILTER_EXPERIMENT_PATH="${record_path}"
  start_server cap-accept "${server_log}" \
    --speculative-dspark-sps-table-path "${SPS_TABLE_PATH}"

  # shellcheck disable=SC2206
  extra_bench_args=(${EXTRA_BENCH_ARGS})
  python -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "http://${HOST}:${PORT}" \
    --model "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-path "${dataset_for_run}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --request-rate "${REQUEST_RATE}" \
    --seed "${seed}" \
    --flush-cache \
    --output-file "${bench_result}" \
    --output-details \
    "${extra_bench_args[@]}" 2>&1 | tee "${bench_log}"

  stop_server
  SERVER_PID=""
  if [[ ! -s "${record_path}" ]]; then
    echo "error: no filter records were written for run ${run}" >&2
    exit 1
  fi
  records+=("${record_path}")
done

python "${REPO_ROOT}/scripts/analyze_dspark_filter.py" \
  "${records[@]}" \
  --margin "${EDGE_MARGIN}" \
  --output "${RESULTS_ROOT}/reports/summary.json"

echo "Report: ${RESULTS_ROOT}/reports/summary.json"
