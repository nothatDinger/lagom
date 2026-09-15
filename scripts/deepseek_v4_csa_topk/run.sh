#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:?set MODEL_PATH}" "${DATASET_PATH:?set DATASET_PATH}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULTS_ROOT=${RESULTS_DIR:-"$ROOT/results/deepseek_v4_csa_topk"}
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-30000}; TP_SIZE=${TP_SIZE:-8}
NUM_PROMPTS=${NUM_PROMPTS:-100}; DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-5}
DETERMINISTIC_INFERENCE=${DETERMINISTIC_INFERENCE:-1}
case "$DETERMINISTIC_INFERENCE" in
  1|true|TRUE|yes|YES) deterministic_label=det_on; deterministic_args=(--enable-deterministic-inference) ;;
  0|false|FALSE|no|NO) deterministic_label=det_off; deterministic_args=() ;;
  *) echo "DETERMINISTIC_INFERENCE must be 1/0, true/false, or yes/no" >&2; exit 2 ;;
esac
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
run_name="${RUN_TIMESTAMP}_${deterministic_label}"
OUT="$RESULTS_ROOT/$run_name"
collision=1
while [[ -e "$OUT" ]]; do
  OUT="$RESULTS_ROOT/${run_name}_$collision"
  ((collision += 1))
done
mkdir -p "$OUT"
printf 'timestamp_utc=%s\ndeterministic_inference=%s\nmodel_path=%s\ndataset_path=%s\n' \
  "$RUN_TIMESTAMP" "$DETERMINISTIC_INFERENCE" "$MODEL_PATH" "$DATASET_PATH" >"$OUT/run_config.txt"
ln -sfn "$(basename "$OUT")" "$RESULTS_ROOT/latest"
python3 "$ROOT/scripts/deepseek_v4_csa_topk/sample_sharegpt.py" --input "$DATASET_PATH" --output "$OUT/sharegpt_first_${NUM_PROMPTS}.json" --count "$NUM_PROMPTS"

server_pid=""
cleanup() { [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

run_server() {
  local k=$1 mode=$2 dir="$OUT/k$1"
  mkdir -p "$dir"
  local trace_env=() trace_args=()
  if [[ "$mode" == trace ]]; then
    trace_env=("SGLANG_HISPARSE_H2D_TRACE_PATH=$dir/h2d_trace")
    trace_args=(--disable-cuda-graph)
  fi
  env "${trace_env[@]}" python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --tp "$TP_SIZE" --host "$HOST" --port "$PORT" \
    "${deterministic_args[@]}" \
    --enable-hisparse --hisparse-config "{\"top_k\":$k,\"device_buffer_size\":$(( (DSPARK_BLOCK_SIZE + 1) * k )),\"host_to_device_ratio\":${HOST_TO_DEVICE_RATIO:-5}}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE" "${trace_args[@]}" ${SERVER_EXTRA_ARGS:-} \
    >"$dir/server_${mode}.log" 2>"$dir/server_${mode}.err" &
  server_pid=$!
  for _ in $(seq 1 "${SERVER_WAIT_POLLS:-1800}"); do
    curl -fsS "http://$HOST:$PORT/health" >/dev/null && return
    kill -0 "$server_pid" 2>/dev/null || { cat "$dir/server_${mode}.err" >&2; return 1; }
    sleep 2
  done
  echo "server readiness timeout" >&2; return 1
}

benchmark() {
  local k=$1 mode=$2 output_args=()
  [[ "$mode" != perf ]] || output_args=(--output-file "$OUT/k$k/benchmark.jsonl" --output-details)
  python3 -m sglang.benchmark.serving --backend sglang --host "$HOST" --port "$PORT" \
    --dataset-name sharegpt --dataset-path "$OUT/sharegpt_first_${NUM_PROMPTS}.json" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --seed 0 "${output_args[@]}" \
    >"$OUT/k$k/client_${mode}.log" 2>"$OUT/k$k/client_${mode}.err"
}

for k in 512 1024 2048 4096; do
  rm -f "$OUT/k$k/h2d_trace.tp"*.jsonl
  run_server "$k" perf; benchmark "$k" perf; cleanup; wait "$server_pid" 2>/dev/null || true; server_pid=""
  run_server "$k" trace; benchmark "$k" trace; cleanup; wait "$server_pid" 2>/dev/null || true; server_pid=""
done
python3 "$ROOT/scripts/deepseek_v4_csa_topk/analyze.py" --results-dir "$OUT"
