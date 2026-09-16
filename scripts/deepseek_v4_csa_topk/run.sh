#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULTS_DIR=${RESULTS_DIR:-"$ROOT/results"}
RESULTS_ROOT="$RESULTS_DIR/deepseek_v4_csa_topk"
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-30000}; TP_SIZE=${TP_SIZE:-8}
NUM_PROMPTS=${NUM_PROMPTS:-1}; DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-5}
DATASET_NAME=${DATASET_NAME:-random}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-110000}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-512}
if [[ "$DATASET_NAME" == sharegpt ]]; then
  : "${DATASET_PATH:?set DATASET_PATH when DATASET_NAME=sharegpt}"
fi
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-1}
DETERMINISTIC_INFERENCE=${DETERMINISTIC_INFERENCE:-0}
case "$DETERMINISTIC_INFERENCE" in
  1|true|TRUE|yes|YES)
    echo "DETERMINISTIC_INFERENCE=1 is unsupported for DeepSeek V4: the model requires attention_backend=dsv4, but dsv4 is not a supported deterministic attention backend in this SGLang version." >&2
    exit 2
    ;;
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
printf 'timestamp_utc=%s\ndeterministic_inference=%s\nmodel_path=%s\ndataset_path=%s\nresults_dir=%s\nmoe_runner_backend=%s\nmem_fraction_static=%s\ncuda_graph_max_bs_decode=%s\n' \
  "$RUN_TIMESTAMP" "$DETERMINISTIC_INFERENCE" "$MODEL_PATH" "$DATASET_PATH" \
  "$RESULTS_DIR" "$MOE_RUNNER_BACKEND" "$MEM_FRACTION_STATIC" \
  "$CUDA_GRAPH_MAX_BS_DECODE" \
  >"$OUT/run_config.txt"
ln -sfn "$(basename "$OUT")" "$RESULTS_ROOT/latest"
benchmark_dataset_args=(--dataset-name random --random-input-len "$RANDOM_INPUT_LEN" --random-output-len "$RANDOM_OUTPUT_LEN" --random-range-ratio 0)
if [[ "$DATASET_NAME" == sharegpt ]]; then
  python3 "$ROOT/scripts/deepseek_v4_csa_topk/sample_sharegpt.py" --input "$DATASET_PATH" --output "$OUT/sharegpt_first_${NUM_PROMPTS}.json" --count "$NUM_PROMPTS"
  benchmark_dataset_args=(--dataset-name sharegpt --dataset-path "$OUT/sharegpt_first_${NUM_PROMPTS}.json")
elif [[ "$DATASET_NAME" != random ]]; then
  echo "DATASET_NAME must be random or sharegpt" >&2
  exit 2
fi

server_pid=""
current_k="-"; current_mode="setup"; current_state="starting"
write_state() {
  local tmp="$OUT/run_state.tmp"
  printf 'state=%s\nk=%s\nmode=%s\nserver_pid=%s\nupdated_at_utc=%s\n' \
    "$current_state" "$current_k" "$current_mode" "${server_pid:--}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$tmp"
  mv "$tmp" "$OUT/run_state.env"
}
cleanup() { [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true; }
on_exit() {
  local status=$?
  cleanup
  if (( status == 0 )); then current_state=complete; else current_state=failed; fi
  write_state
}
trap on_exit EXIT
trap 'exit 130' INT TERM
write_state

run_server() {
  local k=$1 mode=$2 dir="$OUT/k$1"
  current_k=$k; current_mode=$mode; current_state=server_starting; write_state
  mkdir -p "$dir"
  local trace_env=() trace_args=()
  if [[ "$mode" == trace ]]; then
    trace_env=("SGLANG_HISPARSE_H2D_TRACE_PATH=$dir/h2d_trace")
    trace_args=(--disable-cuda-graph)
  fi
  env "${trace_env[@]}" python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" --tp "$TP_SIZE" --host "$HOST" --port "$PORT" \
    --moe-runner-backend "$MOE_RUNNER_BACKEND" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE" \
    "${deterministic_args[@]}" \
    --enable-hisparse --disable-radix-cache \
    --hisparse-config "{\"top_k\":$k,\"device_buffer_size\":$(( (DSPARK_BLOCK_SIZE + 1) * k )),\"host_to_device_ratio\":${HOST_TO_DEVICE_RATIO:-5}}" \
    --speculative-algorithm DSPARK \
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE" "${trace_args[@]}" ${SERVER_EXTRA_ARGS:-} \
    >"$dir/server_${mode}.log" 2>"$dir/server_${mode}.err" &
  server_pid=$!
  write_state
  for _ in $(seq 1 "${SERVER_WAIT_POLLS:-1800}"); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null; then
      current_state=server_ready; write_state; return
    fi
    kill -0 "$server_pid" 2>/dev/null || { cat "$dir/server_${mode}.err" >&2; return 1; }
    sleep 2
  done
  echo "server readiness timeout" >&2; return 1
}

benchmark() {
  local k=$1 mode=$2 output_file
  if [[ "$mode" == perf ]]; then
    output_file="$OUT/k$k/benchmark.jsonl"
  else
    output_file="$OUT/k$k/benchmark_trace.jsonl"
  fi
  rm -f "$output_file"
  current_k=$k; current_mode=$mode; current_state=benchmark_running; write_state
  python3 -m sglang.benchmark.serving --backend sglang --host "$HOST" --port "$PORT" \
    "${benchmark_dataset_args[@]}" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency 1 --seed 0 \
    --output-file "$output_file" --output-details \
    >"$OUT/k$k/client_${mode}.log" 2>"$OUT/k$k/client_${mode}.err"
}

for k in 512 1024 2048 4096; do
  rm -f "$OUT/k$k/h2d_trace.tp"*.jsonl
  run_server "$k" perf; benchmark "$k" perf; cleanup; wait "$server_pid" 2>/dev/null || true; server_pid=""
  run_server "$k" trace; benchmark "$k" trace; cleanup; wait "$server_pid" 2>/dev/null || true; server_pid=""
done
current_state=analyzing; current_k="-"; current_mode="analysis"; server_pid=""; write_state
python3 "$ROOT/scripts/deepseek_v4_csa_topk/analyze.py" --results-dir "$OUT"
