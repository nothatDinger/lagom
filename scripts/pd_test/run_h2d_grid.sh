#!/usr/bin/env bash
# Run INSIDE the existing gpuq allocation of physical GPU4/5.
set -euo pipefail
if [[ $# -ne 2 || ( "$1" != smoke && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s {smoke|main} /new/output/dir\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set LAGOM_PY to the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" ]]; then
  printf 'LAGOM_PY must be an executable absolute path\n' >&2
  exit 2
fi
H2D_GRID_MODE="$1"
H2D_GRID_OUT="$2"
if [[ "$H2D_GRID_OUT" != /* || -e "$H2D_GRID_OUT" ]]; then
  printf 'Use a new absolute output directory: %s\n' "$H2D_GRID_OUT" >&2
  exit 2
fi
H2D_GRID_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_GRID_REPO"
export PYTHONPATH="$H2D_GRID_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_GRID_OUT"
H2D_GRID_ARGS=(
  --copy-grid-control --device-source-control --register-device 1
  --phases compute --profile --output-dir "$H2D_GRID_OUT"
)
if [[ "$H2D_GRID_MODE" == smoke ]]; then
  H2D_GRID_ARGS+=(
    --batch 4 --misses 2 70 --plan-stride 128 --buffer-slots 128
    --source-mib 4 --max-banks 4 --bulk-ce-mib 1
    --epochs 1 --warmup 1 --warm-ms 20
  )
else
  H2D_GRID_ARGS+=(
    --batch 32 --misses 100 --plan-stride 3072 --buffer-slots 3072
    --source-mib 512 --max-banks 64 --bulk-ce-mib 64
    --epochs 3 --warmup 2 --warm-ms 250
  )
fi
if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py "${H2D_GRID_ARGS[@]}" \
  >"$H2D_GRID_OUT/run.log" 2>&1; then
  cat "$H2D_GRID_OUT/grid-compact.txt"
  if ! rg -q 'results_valid=True' "$H2D_GRID_OUT/grid-compact.txt"; then
    printf 'Grid validation incomplete; retain pair.json, trace and run.log\n' >&2
    exit 1
  fi
else
  H2D_GRID_STATUS=$?
  if [[ -f "$H2D_GRID_OUT/grid-compact.txt" ]]; then
    cat "$H2D_GRID_OUT/grid-compact.txt"
  fi
  tail -n 30 "$H2D_GRID_OUT/run.log" >&2
  exit "$H2D_GRID_STATUS"
fi
