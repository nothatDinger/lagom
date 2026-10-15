#!/usr/bin/env bash
# Run inside one existing gpuq allocation of physical GPU4/5.
set -euo pipefail
if [[ $# -ne 1 ]]; then
  printf 'Usage: LAGOM_PY=/absolute/venv/python bash %s /new/absolute/output\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" || "$1" != /* || -e "$1" ]]; then
  printf 'Require executable absolute LAGOM_PY and fresh absolute output directory\n' >&2
  exit 2
fi
H2D_ORDER_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_ORDER_OUT="$1"
cd "$H2D_ORDER_REPO"
export PYTHONPATH="$H2D_ORDER_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
mkdir -p "$H2D_ORDER_OUT"
for H2D_ORDER_PAIR in 4,5 5,4; do
  if [[ "$H2D_ORDER_PAIR" == 4,5 ]]; then
    H2D_ORDER_LABEL=normal
  else
    H2D_ORDER_LABEL=reversed
  fi
  mkdir "$H2D_ORDER_OUT/$H2D_ORDER_LABEL"
  if CUDA_VISIBLE_DEVICES="$H2D_ORDER_PAIR" numactl --cpunodebind=1 --membind=1 \
    "$LAGOM_PY" scripts/pd_test/bench_h2d_device_order.py run \
    --physical-order "$H2D_ORDER_PAIR" --output-dir "$H2D_ORDER_OUT/$H2D_ORDER_LABEL" \
    >"$H2D_ORDER_OUT/$H2D_ORDER_LABEL/run.log" 2>&1; then
    cat "$H2D_ORDER_OUT/$H2D_ORDER_LABEL/order-compact.txt"
  else
    H2D_ORDER_STATUS=$?
    if [[ -f "$H2D_ORDER_OUT/$H2D_ORDER_LABEL/order-compact.txt" ]]; then
      cat "$H2D_ORDER_OUT/$H2D_ORDER_LABEL/order-compact.txt"
    fi
    tail -n 30 "$H2D_ORDER_OUT/$H2D_ORDER_LABEL/run.log" >&2
    exit "$H2D_ORDER_STATUS"
  fi
done
"$LAGOM_PY" scripts/pd_test/bench_h2d_device_order.py summarize \
  --normal "$H2D_ORDER_OUT/normal/order.json" \
  --reversed "$H2D_ORDER_OUT/reversed/order.json" --output-dir "$H2D_ORDER_OUT" \
  >"$H2D_ORDER_OUT/summary.log" 2>&1
cat "$H2D_ORDER_OUT/order-compact.txt"
