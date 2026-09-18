#!/usr/bin/env bash
set -Eeuo pipefail

: "${MODEL_PATH:?set MODEL_PATH}"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULTS_DIR=${RESULTS_DIR:-"$ROOT/results"}
RESULTS_ROOT="$RESULTS_DIR/deepseek_v4_csa_topk_h2d_batchsize"
HOST=${HOST:-127.0.0.1}; PORT=${PORT:-30000}; TP_SIZE=${TP_SIZE:-8}
BATCH_SIZES=${BATCH_SIZES:-"1 4 8 16 32"}; readonly TOP_K=512; DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-5}
read -r -a batch_sizes <<<"$BATCH_SIZES"
(( ${#batch_sizes[@]} > 0 )) || { echo "BATCH_SIZES must contain at least one positive integer" >&2; exit 2; }
max_batch_size=0
for batch_size in "${batch_sizes[@]}"; do
  [[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch size: $batch_size" >&2; exit 2; }
  (( batch_size > max_batch_size )) && max_batch_size=$batch_size
done
DATASET_NAME=${DATASET_NAME:-random}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-110000}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-512}
LONGBENCH_OUTPUT_LEN=${LONGBENCH_OUTPUT_LEN:-512}
REQUEST_INPUT_LENGTH_LIMIT_MODE=${REQUEST_INPUT_LENGTH_LIMIT_MODE:-none}
case "$REQUEST_INPUT_LENGTH_LIMIT_MODE" in
  none|filter|truncate) ;;
  *) echo "REQUEST_INPUT_LENGTH_LIMIT_MODE must be none, filter, or truncate" >&2; exit 2 ;;
esac
if [[ "$DATASET_NAME" == sharegpt ]]; then
  : "${DATASET_PATH:?set DATASET_PATH when DATASET_NAME=sharegpt}"
elif [[ "$DATASET_NAME" == longbench ]]; then
  : "${DATASET_PATH:?set DATASET_PATH when DATASET_NAME=longbench}"
elif [[ "$DATASET_NAME" == longbench_v2 || "$DATASET_NAME" == longbench-v2 ]]; then
  DATASET_PATH=${DATASET_PATH:-/home/jovyan/td69032/LongBench-v2}
fi
MOE_RUNNER_BACKEND=${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.85}
CUDA_GRAPH_MAX_BS_DECODE=${CUDA_GRAPH_MAX_BS_DECODE:-32}
SERVER_RESTART_DELAY=${SERVER_RESTART_DELAY:-10}
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
printf 'timestamp_utc=%s\ndeterministic_inference=%s\nmodel_path=%s\ndataset_name=%s\ndataset_path=%s\nrandom_input_len=%s\nrandom_output_len=%s\nlongbench_output_len=%s\nrequest_input_length_limit_mode=%s\ntop_k=%s\nbatch_sizes=%s\nserver_restart_delay=%s\nresults_dir=%s\nmoe_runner_backend=%s\nmem_fraction_static=%s\ncuda_graph_max_bs_decode=%s\n' \
  "$RUN_TIMESTAMP" "$DETERMINISTIC_INFERENCE" "$MODEL_PATH" "$DATASET_NAME" \
  "${DATASET_PATH:-}" "$RANDOM_INPUT_LEN" "$RANDOM_OUTPUT_LEN" "$LONGBENCH_OUTPUT_LEN" \
  "$REQUEST_INPUT_LENGTH_LIMIT_MODE" "$TOP_K" "$BATCH_SIZES" "$SERVER_RESTART_DELAY" "$RESULTS_DIR" "$MOE_RUNNER_BACKEND" "$MEM_FRACTION_STATIC" \
  "$CUDA_GRAPH_MAX_BS_DECODE" \
  >"$OUT/run_config.txt"
ln -sfn "$(basename "$OUT")" "$RESULTS_ROOT/latest"
benchmark_dataset_args=(--dataset-name random --random-input-len "$RANDOM_INPUT_LEN" --random-output-len "$RANDOM_OUTPUT_LEN" --random-range-ratio 0)
if [[ "$DATASET_NAME" == random && -n "${DATASET_PATH:-}" ]]; then
  benchmark_dataset_args+=(--dataset-path "$DATASET_PATH")
elif [[ "$DATASET_NAME" == sharegpt ]]; then
  python3 "$ROOT/scripts/deepseek_v4_csa_topk_h2d_batchsize/sample_sharegpt.py" --input "$DATASET_PATH" --output "$OUT/sharegpt_first_${max_batch_size}.json" --count "$max_batch_size"
  benchmark_dataset_args=(--dataset-name sharegpt --dataset-path "$OUT/sharegpt_first_${max_batch_size}.json")
elif [[ "$DATASET_NAME" == longbench || "$DATASET_NAME" == longbench_v2 || "$DATASET_NAME" == longbench-v2 ]]; then
  longbench_variant=$DATASET_NAME
  [[ "$longbench_variant" != longbench-v2 ]] || longbench_variant=longbench_v2
  prepared_dataset="$OUT/${longbench_variant}_first_${max_batch_size}.json"
  prepare_count=$max_batch_size
  # The client needs later rows available in order to replace samples rejected
  # by its 128K input-token filter.
  [[ "$REQUEST_INPUT_LENGTH_LIMIT_MODE" != filter ]] || prepare_count=0
  python3 "$ROOT/scripts/deepseek_v4_csa_topk_h2d_batchsize/prepare_longbench.py" \
    --input "$DATASET_PATH" --output "$prepared_dataset" \
    --variant "$longbench_variant" --count "$prepare_count"
  benchmark_dataset_args=(--dataset-name sharegpt --dataset-path "$prepared_dataset" --sharegpt-output-len "$LONGBENCH_OUTPUT_LEN")
  if [[ "$REQUEST_INPUT_LENGTH_LIMIT_MODE" == filter ]]; then
    # ShareGPT's loader keeps scanning after an oversized row, so apply the
    # admission policy before selecting the requested batch rather than letting the
    # server reject (for example) LongBench-v2's 273K-token first sample.
    benchmark_dataset_args+=(--sharegpt-context-len "$((REQUEST_INPUT_LENGTH_LIMIT + LONGBENCH_OUTPUT_LEN))")
  fi
elif [[ "$DATASET_NAME" != random ]]; then
  echo "DATASET_NAME must be random, sharegpt, longbench, or longbench_v2" >&2
  exit 2
fi

server_pid=""
current_batch_size="-"; current_mode="setup"; current_state="starting"
write_state() {
  local tmp="$OUT/run_state.tmp"
  printf 'state=%s\nbatch_size=%s\nmode=%s\nserver_pid=%s\nupdated_at_utc=%s\n' \
    "$current_state" "$current_batch_size" "$current_mode" "${server_pid:--}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$tmp"
  mv "$tmp" "$OUT/run_state.env"
}
cleanup() { [[ -z "$server_pid" ]] || kill "$server_pid" 2>/dev/null || true; }
stop_server() {
  cleanup
  [[ -z "$server_pid" ]] || wait "$server_pid" 2>/dev/null || true
  server_pid=""
  # launch_server can otherwise race the previous listener's teardown on the
  # scheduler-assigned PORT when switching between perf and trace.
  sleep "$SERVER_RESTART_DELAY"
}
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
  local batch_size=$1 mode=$2 dir="$OUT/bs$1"
  current_batch_size=$batch_size; current_mode=$mode; current_state=server_starting; write_state
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
    --request-input-length-limit-mode "$REQUEST_INPUT_LENGTH_LIMIT_MODE" \
    "${deterministic_args[@]}" \
    --enable-hisparse --disable-radix-cache \
    --hisparse-config "{\"top_k\":$TOP_K,\"device_buffer_size\":$(( (DSPARK_BLOCK_SIZE + 1) * TOP_K )),\"host_to_device_ratio\":${HOST_TO_DEVICE_RATIO:-5}}" \
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
  local batch_size=$1 mode=$2 output_file
  if [[ "$mode" == perf ]]; then
    output_file="$OUT/bs$batch_size/benchmark.jsonl"
  else
    output_file="$OUT/bs$batch_size/benchmark_trace.jsonl"
  fi
  rm -f "$output_file"
  current_batch_size=$batch_size; current_mode=$mode; current_state=benchmark_running; write_state
  python3 -m sglang.benchmark.serving --backend sglang --host "$HOST" --port "$PORT" \
    "${benchmark_dataset_args[@]}" \
    --num-prompts "$batch_size" --max-concurrency "$batch_size" --seed 0 \
    --output-file "$output_file" --output-details \
    >"$OUT/bs$batch_size/client_${mode}.log" 2>"$OUT/bs$batch_size/client_${mode}.err"
}

for batch_size in "${batch_sizes[@]}"; do
  rm -f "$OUT/bs$batch_size/h2d_trace.tp"*.jsonl
  run_server "$batch_size" perf; benchmark "$batch_size" perf; stop_server
  run_server "$batch_size" trace; benchmark "$batch_size" trace; stop_server
done
current_state=analyzing; current_batch_size="-"; current_mode="analysis"; server_pid=""; write_state
python3 "$ROOT/scripts/deepseek_v4_csa_topk_h2d_batchsize/analyze.py" --results-dir "$OUT"
