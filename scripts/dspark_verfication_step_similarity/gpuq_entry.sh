#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
[[ "$RESULTS_DIR" == /* ]] || RESULTS_DIR="$ROOT/$RESULTS_DIR"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_DIR/scripts/dspark_verfication_step_similarity/$RUN_TIMESTAMP"
mkdir -p "$OUT"
printf 'Experiment output: %s\n' "$OUT"

: "${MODEL_PATH:?required}"
: "${PORT:=33464}" "${TP_SIZE:=4}" "${MEM_FRACTION_STATIC:=0.80}"
: "${DATASET_NAME:=random}" "${RANDOM_INPUT_LEN:=128000}"
: "${RANDOM_OUTPUT_LEN:=512}" "${NUM_PROMPTS:=100}"
: "${SERVER_RESTART_DELAY:=5}" "${HOST:=127.0.0.1}"
: "${MOE_RUNNER_BACKEND:=flashinfer_mxfp4}"
: "${HISPARSE_CONFIG:={\"top_k\":2048,\"host_to_device_ratio\":5}}"

export SGLANG_DSPARK_DEBUG_DUMP=core,reqs
export SGLANG_DSPARK_RECORD_VERIFICATION_STEP_SIMILARITY=1
if [[ "${DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
  export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=1
fi

cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

server_args=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --speculative-algorithm DSPARK
  --enable-hisparse
  --hisparse-config "$HISPARSE_CONFIG"
  --disable-radix-cache
  --disable-cuda-graph
  --moe-runner-backend "$MOE_RUNNER_BACKEND"
  --tp-size "$TP_SIZE"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --host "$HOST"
  --port "$PORT"
)
if [[ -n "${SERVER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($SERVER_EXTRA_ARGS)
  server_args+=("${extra_args[@]}")
fi

python3 -m sglang.launch_server "${server_args[@]}" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
BASE_URL="http://$HOST:$PORT"
python3 "$ROOT/scripts/dspark_verfication_step_similarity/dump_records.py" prepare \
  --base-url "$BASE_URL" --restart-delay "$SERVER_RESTART_DELAY"

python3 -m sglang.benchmark.serving \
  --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL_PATH" \
  --dataset-name "$DATASET_NAME" --dataset-path "${DATASET_PATH:-}" \
  --random-input-len "$RANDOM_INPUT_LEN" --random-output-len "$RANDOM_OUTPUT_LEN" \
  --random-range-ratio 0 --num-prompts "$NUM_PROMPTS" \
  --output-file "$OUT/benchmark.jsonl" 2>&1 | tee "$OUT/client.log"

python3 "$ROOT/scripts/dspark_verfication_step_similarity/dump_records.py" dump \
  --base-url "$BASE_URL" --output "$OUT/raw_records.json" \
  --dataset-name "$DATASET_NAME" --count "$NUM_PROMPTS" \
  --input-len "$RANDOM_INPUT_LEN" --output-len "$RANDOM_OUTPUT_LEN"
python3 "$ROOT/scripts/dspark_verfication_step_similarity/analyze.py" \
  --input "$OUT/raw_records.json" --output-dir "$OUT" 2>&1 | tee "$OUT/analysis.log"
