#!/usr/bin/env bash
# Run inside the existing physical GPU4/5 gpuq allocation.
set -euo pipefail
if [[ $# -ne 2 || ( "$1" != check && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s {check|main} /new/absolute/output\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" ]]; then
  printf 'Require executable absolute LAGOM_PY\n' >&2
  exit 2
fi
H2D_LINEAR_MODE="$1"
H2D_LINEAR_OUT="$2"
if [[ "$H2D_LINEAR_OUT" != /* || -e "$H2D_LINEAR_OUT" ]]; then
  printf 'Use a fresh absolute output directory\n' >&2
  exit 2
fi
H2D_LINEAR_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_LINEAR_REPO"
export PYTHONPATH="$H2D_LINEAR_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_LINEAR_OUT"
if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_linear.py "$H2D_LINEAR_MODE" --output-dir "$H2D_LINEAR_OUT" \
  >"$H2D_LINEAR_OUT/run.log" 2>&1; then
  cat "$H2D_LINEAR_OUT/linear-compact.txt"
else
  H2D_LINEAR_STATUS=$?
  if [[ -f "$H2D_LINEAR_OUT/linear-compact.txt" ]]; then
    cat "$H2D_LINEAR_OUT/linear-compact.txt"
  fi
  tail -n 30 "$H2D_LINEAR_OUT/run.log" >&2
  exit "$H2D_LINEAR_STATUS"
fi
