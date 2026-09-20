#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MODEL_PATH:-}" || -z "${DATASET_PATH:-}" ]]; then
  echo "error: MODEL_PATH and DATASET_PATH must be exported by the GPU job" >&2
  exit 2
fi

echo "[DSpark experiment] profiling SPS table"
bash "${SCRIPT_DIR}/profile_sps.sh"

echo "[DSpark experiment] running filter-boundary repetitions"
bash "${SCRIPT_DIR}/run_experiment.sh"
