#!/usr/bin/env bash
# Run INSIDE an allocation of physical GPU4/5. No model server or NCU required.
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
H2D_LAYOUT_MODE="$1"
H2D_LAYOUT_OUT="$2"
if [[ "$H2D_LAYOUT_OUT" != /* || -e "$H2D_LAYOUT_OUT" ]]; then
  printf 'Use a new absolute output directory: %s\n' "$H2D_LAYOUT_OUT" >&2
  exit 2
fi
H2D_LAYOUT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_LAYOUT_REPO"
export PYTHONPATH="$H2D_LAYOUT_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_LAYOUT_OUT"

H2D_LAYOUT_ARGS=(
  --source-layout-control --device-source-control --register-device 1
  --phases compute --profile --output-dir "$H2D_LAYOUT_OUT"
)
if [[ "$H2D_LAYOUT_MODE" == smoke ]]; then
  H2D_LAYOUT_ARGS+=(
    --batch 2 --misses 2 70 --plan-stride 128 --buffer-slots 128
    --source-mib 4 --max-banks 4 --bulk-ce-mib 1
    --epochs 1 --warmup 1 --warm-ms 20
  )
else
  H2D_LAYOUT_ARGS+=(
    --batch 32 --misses 100 1272 --plan-stride 3072 --buffer-slots 3072
    --source-mib 512 --max-banks 8 --bulk-ce-mib 64
    --epochs 2 --warmup 2 --warm-ms 250
  )
fi

if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py "${H2D_LAYOUT_ARGS[@]}" \
  >"$H2D_LAYOUT_OUT/run.log" 2>&1; then
  cat "$H2D_LAYOUT_OUT/layout-compact.txt"
  if ! rg -q 'ready_for_main=True' "$H2D_LAYOUT_OUT/layout-compact.txt"; then
    printf 'Layout validation incomplete; retain run.log and pair.json\n' >&2
    exit 1
  fi
else
  H2D_LAYOUT_STATUS=$?
  if [[ -f "$H2D_LAYOUT_OUT/layout-compact.txt" ]]; then
    cat "$H2D_LAYOUT_OUT/layout-compact.txt"
  fi
  tail -n 30 "$H2D_LAYOUT_OUT/run.log" >&2
  exit "$H2D_LAYOUT_STATUS"
fi
