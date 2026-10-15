#!/usr/bin/env bash
# Existing gpuq allocation of physical GPU4; expose this GPU only.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != --rx-sampler ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s /new/absolute/output [--rx-sampler]\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" || "$1" != /* || -e "$1" ]]; then
  printf 'Require executable absolute LAGOM_PY and fresh absolute output directory\n' >&2
  exit 2
fi
H2D_BULK_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_BULK_OUT="$1"
shift
cd "$H2D_BULK_REPO"
export PYTHONPATH="$H2D_BULK_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p "$H2D_BULK_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_bulk_compare.py --output-dir "$H2D_BULK_OUT" "$@" \
  >"$H2D_BULK_OUT/run.log" 2>&1; then
  cat "$H2D_BULK_OUT/bulk-compact.txt"
else
  H2D_BULK_STATUS=$?
  if [[ -f "$H2D_BULK_OUT/bulk-compact.txt" ]]; then cat "$H2D_BULK_OUT/bulk-compact.txt"; fi
  tail -n 30 "$H2D_BULK_OUT/run.log" >&2
  exit "$H2D_BULK_STATUS"
fi
