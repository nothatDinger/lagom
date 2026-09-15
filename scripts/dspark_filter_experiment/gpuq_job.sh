#!/usr/bin/env bash
# GPUQ payload. Submit this file with your site's gpuq command; do not run the
# server and benchmark as separate GPUQ jobs because they share one allocation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${MODEL_PATH:-}" || -z "${DATASET_PATH:-}" ]]; then
  echo "error: pass MODEL_PATH and DATASET_PATH into the gpuq job environment" >&2
  exit 2
fi

cd "${REPO_ROOT}"
echo "GPU allocation: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Results: ${RESULTS_ROOT:-${REPO_ROOT}/results/dspark_filter_experiment}"

# GPUQ normally starts a non-interactive shell. Activate the serving environment
# here when the queue image does not already contain sglang.
if [[ -n "${EXPERIMENT_ENV_SETUP_SCRIPT:-}" ]]; then
  # shellcheck source=/dev/null
  source "${EXPERIMENT_ENV_SETUP_SCRIPT}"
fi

exec bash "${SCRIPT_DIR}/run_all.sh"
