#!/usr/bin/env bash
# Backward-compatible local entry point; gpuq jobs should call gpuq_entry.sh.
set -Eeuo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PATH="${MODEL_PATH:-${TARGET_MODEL_PATH:-}}"
export NUM_PROMPTS="${NUM_PROMPTS:-${SAMPLE_COUNT:-100}}"
export RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-${MAX_NEW_TOKENS:-512}}"
exec "$DIR/gpuq_entry.sh"
