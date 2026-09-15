#!/usr/bin/env bash
# Shared DSpark filter experiment configuration. MODEL_PATH and DATASET_PATH
# intentionally have no defaults: export them in the calling environment.

export TP_SIZE="${TP_SIZE:-4}"
export PORT="${PORT:-30000}"
export HOST="${HOST:-127.0.0.1}"
export MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
export DATASET_NAME="${DATASET_NAME:-sharegpt}"
export NUM_PROMPTS="${NUM_PROMPTS:-1000}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
export REQUEST_RATE="${REQUEST_RATE:-inf}"
export RUN_COUNT="${RUN_COUNT:-3}"
export BASE_SEED="${BASE_SEED:-42}"
export EDGE_MARGIN="${EDGE_MARGIN:-0.02}"
export SPS_MAX_BATCH_SIZE="${SPS_MAX_BATCH_SIZE:-256}"
export SPS_REPEATS="${SPS_REPEATS:-3}"

# Optional whitespace-separated arguments. Prefer dedicated variables above.
export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"
export EXTRA_BENCH_ARGS="${EXTRA_BENCH_ARGS:-}"
export EXTRA_SPS_ARGS="${EXTRA_SPS_ARGS:-}"
