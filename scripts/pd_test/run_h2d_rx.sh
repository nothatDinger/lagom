#!/usr/bin/env bash
# Run inside the existing GPU4/5 gpuq allocation; existing venv only.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || "$1" != check ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s check /new/output/dir [CUDA_DEVICE=0|1]\nOld smoke/main are paused until the sampler self-check is reviewed.\n' "$0" >&2
  exit 2
fi
H2D_RX_DEVICE="${3:-0}"
if [[ "$H2D_RX_DEVICE" != 0 && "$H2D_RX_DEVICE" != 1 ]]; then
  printf 'CUDA_DEVICE must be 0 (GPU4) or 1 (GPU5)\n' >&2
  exit 2
fi
: "${LAGOM_PY:?Set LAGOM_PY to the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" ]]; then
  printf 'Require executable absolute LAGOM_PY\n' >&2
  exit 2
fi
H2D_RX_OUT="$2"
if [[ "$H2D_RX_OUT" != /* || -e "$H2D_RX_OUT" ]]; then
  printf 'Use a new absolute output directory: %s\n' "$H2D_RX_OUT" >&2
  exit 2
fi
H2D_RX_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_RX_REPO"
export PYTHONPATH="$H2D_RX_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_RX_OUT"
H2D_RX_ARGS=(--output-dir "$H2D_RX_OUT" --sampler-check --check-device "$H2D_RX_DEVICE" --sample-ms 50 --chunk-ms 100
  --epochs 1 --duration-s 2.5 --trim-s .3)
if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_rx.py "${H2D_RX_ARGS[@]}" \
  >"$H2D_RX_OUT/run.log" 2>&1; then
  cat "$H2D_RX_OUT/rx-compact.txt"
  if ! rg -q 'results_valid=True' "$H2D_RX_OUT/rx-compact.txt"; then
    printf 'Rx validation incomplete; retain rx.json and run.log\n' >&2
    exit 1
  fi
else
  H2D_RX_STATUS=$?
  if [[ -f "$H2D_RX_OUT/rx-compact.txt" ]]; then
    cat "$H2D_RX_OUT/rx-compact.txt"
  fi
  tail -n 30 "$H2D_RX_OUT/run.log" >&2
  exit "$H2D_RX_STATUS"
fi
