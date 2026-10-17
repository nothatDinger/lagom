#!/usr/bin/env bash
# One GPU4 job: full correctness gate, then balanced whole-pipeline timing.
set -euo pipefail
if [[ $# -ne 1 || "$1" != /* || -e "$1" ]]; then
  printf 'Usage: LAGOM_PY=/absolute/python bash %s /new/absolute/output\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set existing venv Python}"
[[ "$LAGOM_PY" == /* && -x "$LAGOM_PY" ]] || exit 2
command -v numactl >/dev/null
H2D_PIPE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_PIPE_OUT="$1"
cd "$H2D_PIPE_ROOT"
export PYTHONPATH="$H2D_PIPE_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_PIPE_OUT"
for H2D_PIPE_MODE in check main; do
  if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
    "$LAGOM_PY" scripts/pd_test/bench_hisparse_pr40670_pipeline.py \
    --mode "$H2D_PIPE_MODE" --output-dir "$H2D_PIPE_OUT/$H2D_PIPE_MODE/result" \
    >"$H2D_PIPE_OUT/$H2D_PIPE_MODE.log" 2>&1; then
    cat "$H2D_PIPE_OUT/$H2D_PIPE_MODE/result/pipeline-compact.txt"
  else
    H2D_PIPE_RC=$?
    [[ ! -f "$H2D_PIPE_OUT/$H2D_PIPE_MODE/result/pipeline-compact.txt" ]] || cat "$H2D_PIPE_OUT/$H2D_PIPE_MODE/result/pipeline-compact.txt"
    tail -n 30 "$H2D_PIPE_OUT/$H2D_PIPE_MODE.log" >&2
    exit "$H2D_PIPE_RC"
  fi
done
