#!/usr/bin/env bash
# Run inside a gpuq allocation of physical GPU4/5, using the existing venv.
set -euo pipefail
if [[ $# -ne 3 || ( "$1" != smoke && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s {smoke|main} /iterator-check-v2 /new/output/dir\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set LAGOM_PY to the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" ]]; then
  printf 'LAGOM_PY must be an executable absolute path\n' >&2
  exit 2
fi
H2D_ITER_PERF_MODE="$1"
H2D_ITER_EVIDENCE="$2"
H2D_ITER_PERF_OUT="$3"
if [[ "$H2D_ITER_EVIDENCE" != /* || ! -f "$H2D_ITER_EVIDENCE/check/iterator.json" || ! -f "$H2D_ITER_EVIDENCE/codegen/codegen.json" ]]; then
  printf 'Require existing v2 check/iterator.json and codegen/codegen.json\n' >&2
  exit 2
fi
if [[ "$H2D_ITER_PERF_OUT" != /* || -e "$H2D_ITER_PERF_OUT" ]]; then
  printf 'Use a new absolute output directory: %s\n' "$H2D_ITER_PERF_OUT" >&2
  exit 2
fi
H2D_ITER_PERF_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_ITER_PERF_REPO"
export PYTHONPATH="$H2D_ITER_PERF_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_ITER_PERF_OUT"
H2D_ITER_PERF_ARGS=(
  --copy-iterator-control --device-source-control --register-device 1
  --iterator-evidence "$H2D_ITER_EVIDENCE" --phases compute --profile
  --output-dir "$H2D_ITER_PERF_OUT"
)
if [[ "$H2D_ITER_PERF_MODE" == smoke ]]; then
  H2D_ITER_PERF_ARGS+=(
    --batch 2 --misses 2 70 --plan-stride 128 --buffer-slots 128
    --source-mib 4 --max-banks 4 --bulk-ce-mib 1
    --epochs 1 --warmup 1 --warm-ms 20
  )
else
  H2D_ITER_PERF_ARGS+=(
    --batch 32 --misses 100 --plan-stride 3072 --buffer-slots 3072
    --source-mib 512 --max-banks 64 --bulk-ce-mib 64
    --epochs 3 --warmup 2 --warm-ms 250
  )
fi
if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py "${H2D_ITER_PERF_ARGS[@]}" \
  >"$H2D_ITER_PERF_OUT/run.log" 2>&1; then
  cat "$H2D_ITER_PERF_OUT/iterator-perf-compact.txt"
  if ! rg -q 'results_valid=True' "$H2D_ITER_PERF_OUT/iterator-perf-compact.txt"; then
    printf 'Iterator performance validation incomplete; retain pair.json and run.log\n' >&2
    exit 1
  fi
else
  H2D_ITER_PERF_STATUS=$?
  if [[ -f "$H2D_ITER_PERF_OUT/iterator-perf-compact.txt" ]]; then
    cat "$H2D_ITER_PERF_OUT/iterator-perf-compact.txt"
  fi
  tail -n 30 "$H2D_ITER_PERF_OUT/run.log" >&2
  exit "$H2D_ITER_PERF_STATUS"
fi
