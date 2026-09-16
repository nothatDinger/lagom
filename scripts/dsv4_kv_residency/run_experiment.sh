#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$ROOT/$RESULTS_DIR"
fi
OUT="$RESULTS_DIR/dsv4_kv_residency"
mkdir -p "$OUT"
: "${MODEL_PATH:?required}" "${DATASET_PATH:?required}"
: "${HOST:=127.0.0.1}" "${PORT:=30000}" "${TP_SIZE:=8}"
: "${MOE_RUNNER_BACKEND:=flashinfer_mxfp4}"
: "${DATASET_NAME:=sharegpt}" "${DATASET_SAMPLING:=head}"
: "${DATASET_SAMPLE_SIZE:=${NUM_PROMPTS:-100}}" "${MEM_FRACTION_STATIC:=0.78}"
: "${HISPARSE_CONFIG:={\"top_k\":2048,\"host_to_device_ratio\":5}}"
[[ "$DATASET_NAME" == "sharegpt" ]] || { echo "Only DATASET_NAME=sharegpt is supported" >&2; exit 2; }
case "$DATASET_SAMPLING" in
  head) sample_method=first ;;
  random) sample_method=random ;;
  *) echo "DATASET_SAMPLING must be head or random" >&2; exit 2 ;;
esac
export SGLANG_DSPARK_DEBUG_DUMP=core,reqs
export SGLANG_DSPARK_RECORD_KV_RESIDENCY=1
SERVER_LOG="$OUT/server.log"; CLIENT_LOG="$OUT/client.log"
cleanup() {
  [[ -z "${SERVER_PID:-}" ]] && return
  kill "$SERVER_PID" 2>/dev/null || return
  local stop_timeout="${SERVER_STOP_TIMEOUT_SECONDS:-0}"
  if (( stop_timeout <= 0 )); then
    wait "$SERVER_PID" 2>/dev/null || true
    return
  fi
  local deadline=$((SECONDS + stop_timeout))
  while kill -0 "$SERVER_PID" 2>/dev/null && (( SECONDS < deadline )); do sleep 1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then kill -KILL "$SERVER_PID" 2>/dev/null || true; fi
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
# The 0731 checkpoint bundles its DSpark head. HiCache and radix prefix caching
# must remain disabled while HiSparse owns the host/device KV hierarchy.
if [[ " ${SERVER_EXTRA_ARGS:-} " == *" --enable-hierarchical-cache "* ||
      " ${SERVER_EXTRA_ARGS:-} " == *" --enable-lmcache "* ]]; then
  echo "SERVER_EXTRA_ARGS must not enable HiCache when HiSparse is active" >&2
  exit 2
fi
server_args=(
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --speculative-algorithm DSPARK
  --enable-hisparse
  --hisparse-config "$HISPARSE_CONFIG"
  --disable-radix-cache
  --disable-cuda-graph
  --moe-runner-backend "$MOE_RUNNER_BACKEND"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --tp-size "$TP_SIZE"
  --host "$HOST"
  --port "$PORT"
)
if [[ "${DETERMINISTIC_INFERENCE:-0}" == "1" ]]; then
  server_args+=(--enable-deterministic-inference)
fi
if [[ -n "${DSPARK_BLOCK_SIZE:-}" ]]; then
  server_args+=(--speculative-dspark-block-size "$DSPARK_BLOCK_SIZE")
fi
if [[ -n "${SERVER_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206 # Deliberate shell-style extra argument interface.
  extra_args=($SERVER_EXTRA_ARGS)
  server_args+=("${extra_args[@]}")
fi
# No timeout is used: gpuq owns job lifetime.
python3 -m sglang.launch_server "${server_args[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
python3 "$ROOT/scripts/dsv4_kv_residency/collect.py" \
  --base-url "http://$HOST:$PORT" --dataset "$DATASET_PATH" \
  --sample-count "$DATASET_SAMPLE_SIZE" --sample-method "$sample_method" \
  --server-ready-timeout-seconds "${SERVER_READY_TIMEOUT_SECONDS:-0}" \
  --seed "${RANDOM_SEED:-0}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --output "$OUT/raw_records.json" 2>&1 | tee "$CLIENT_LOG"
python3 "$ROOT/scripts/dsv4_kv_residency/analyze.py" \
  --input "$OUT/raw_records.json" --output-dir "$OUT" 2>&1 | tee "$OUT/analysis.log"
