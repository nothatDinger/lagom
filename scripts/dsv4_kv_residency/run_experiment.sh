#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$ROOT/$RESULTS_DIR"
fi
OUT="$RESULTS_DIR/dsv4_kv_residency"
mkdir -p "$OUT"
: "${TARGET_MODEL_PATH:?required}" "${DATASET_PATH:?required}"
: "${HOST:=127.0.0.1}" "${PORT:=30000}" "${TP_SIZE:=8}"
: "${MOE_RUNNER_BACKEND:=flashinfer_mxfp4}"
: "${HISPARSE_CONFIG:={\"top_k\":2048,\"host_to_device_ratio\":5}}"
export SGLANG_DSPARK_DEBUG_DUMP=core,reqs
export SGLANG_DSPARK_RECORD_KV_RESIDENCY=1
SERVER_LOG="$OUT/server.log"; CLIENT_LOG="$OUT/client.log"
cleanup() { [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
# The 0731 checkpoint bundles its DSpark head. HiCache and radix prefix caching
# must remain disabled while HiSparse owns the host/device KV hierarchy.
if [[ " ${SERVER_EXTRA_ARGS:-} " == *" --enable-hierarchical-cache "* ||
      " ${SERVER_EXTRA_ARGS:-} " == *" --enable-lmcache "* ]]; then
  echo "SERVER_EXTRA_ARGS must not enable HiCache when HiSparse is active" >&2
  exit 2
fi
server_args=(
  --model-path "$TARGET_MODEL_PATH"
  --trust-remote-code
  --speculative-algorithm DSPARK
  --enable-hisparse
  --hisparse-config "$HISPARSE_CONFIG"
  --disable-radix-cache
  --disable-cuda-graph
  --moe-runner-backend "$MOE_RUNNER_BACKEND"
  --tp-size "$TP_SIZE"
  --host "$HOST"
  --port "$PORT"
)
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
  --dataset-name "${DATASET_NAME:-sharegpt}" \
  --sample-count "${SAMPLE_COUNT:-100}" --sample-method "${SAMPLE_METHOD:-first}" \
  --seed "${RANDOM_SEED:-0}" --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --output "$OUT/raw_records.json" 2>&1 | tee "$CLIENT_LOG"
python3 "$ROOT/scripts/dsv4_kv_residency/analyze.py" \
  --input "$OUT/raw_records.json" --output-dir "$OUT" 2>&1 | tee "$OUT/analysis.log"
