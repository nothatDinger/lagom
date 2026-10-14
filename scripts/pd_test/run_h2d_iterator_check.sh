#!/usr/bin/env bash
# Run inside the existing gpuq allocation of physical GPU4/5.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s /new/output/dir\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set LAGOM_PY to the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" ]]; then
  printf 'LAGOM_PY must be an executable absolute path\n' >&2
  exit 2
fi
H2D_ITER_OUT="$1"
if [[ "$H2D_ITER_OUT" != /* || -e "$H2D_ITER_OUT" ]]; then
  printf 'Use a new absolute output directory: %s\n' "$H2D_ITER_OUT" >&2
  exit 2
fi
H2D_ITER_TOOL="${H2D_CUOBJDUMP:-cuobjdump}"
if ! command -v "$H2D_ITER_TOOL" >/dev/null || ! command -v numactl >/dev/null; then
  printf 'Require existing cuobjdump and numactl; H2D_CUOBJDUMP may select an installed tool\n' >&2
  exit 2
fi
H2D_ITER_TOOL="$(command -v "$H2D_ITER_TOOL")"
H2D_ITER_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_ITER_REPO"
export PYTHONPATH="$H2D_ITER_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_ITER_OUT"

if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/check_h2d_iterator.py \
  --output-dir "$H2D_ITER_OUT/check" >"$H2D_ITER_OUT/check.log" 2>&1; then
  cat "$H2D_ITER_OUT/check/iterator-compact.txt"
else
  H2D_ITER_STATUS=$?
  if [[ -f "$H2D_ITER_OUT/check/iterator-compact.txt" ]]; then
    cat "$H2D_ITER_OUT/check/iterator-compact.txt"
  fi
  tail -n 30 "$H2D_ITER_OUT/check.log" >&2
  exit "$H2D_ITER_STATUS"
fi

if "$LAGOM_PY" scripts/pd_test/collect_h2d_iterator_codegen.py \
  --report "$H2D_ITER_OUT/check/iterator.json" \
  --output-dir "$H2D_ITER_OUT/codegen" --cuobjdump "$H2D_ITER_TOOL" \
  >"$H2D_ITER_OUT/codegen.log" 2>&1; then
  cat "$H2D_ITER_OUT/codegen/codegen-compact.txt"
else
  H2D_ITER_STATUS=$?
  tail -n 30 "$H2D_ITER_OUT/codegen.log" >&2
  exit "$H2D_ITER_STATUS"
fi
printf 'Correctness and extraction finished. Return both compact reports for SASS review; no performance run in this step.\n'
