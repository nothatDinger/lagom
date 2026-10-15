#!/usr/bin/env bash
# Existing physical GPU4/5 allocation; check includes offline cuobjdump extraction.
set -euo pipefail
if [[ $# -lt 2 || ( "$1" != check && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/venv/python bash %s check /new/output OR main /new/output /passed/codegen.json\n' "$0" >&2
  exit 2
fi
if [[ ( "$1" == check && $# -ne 2 ) || ( "$1" == main && $# -ne 3 ) ]]; then
  printf 'check requires 2 arguments; main requires 3\n' >&2
  exit 2
fi
: "${LAGOM_PY:?Set the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" || "$2" != /* || -e "$2" ]]; then
  printf 'Require executable absolute LAGOM_PY and fresh absolute output directory\n' >&2
  exit 2
fi
H2D_DIRECT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_DIRECT_OUT="$2"
H2D_DIRECT_ARGS=("$1" --direct-index-control --output-dir "$H2D_DIRECT_OUT")
if [[ "$1" == main ]]; then
  if [[ "$3" != /* || ! -r "$3" ]]; then
    printf 'Require readable absolute codegen.json from passed check\n' >&2
    exit 2
  fi
  H2D_DIRECT_ARGS+=(--codegen-json "$3")
fi
cd "$H2D_DIRECT_REPO"
export PYTHONPATH="$H2D_DIRECT_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_DIRECT_OUT"
if CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_kv_bridge.py "${H2D_DIRECT_ARGS[@]}" \
  >"$H2D_DIRECT_OUT/run.log" 2>&1; then
  cat "$H2D_DIRECT_OUT/bridge-compact.txt"
else
  H2D_DIRECT_STATUS=$?
  if [[ -f "$H2D_DIRECT_OUT/bridge-compact.txt" ]]; then cat "$H2D_DIRECT_OUT/bridge-compact.txt"; fi
  tail -n 30 "$H2D_DIRECT_OUT/run.log" >&2
  exit "$H2D_DIRECT_STATUS"
fi
if [[ "$1" == check ]]; then
  if "$LAGOM_PY" scripts/pd_test/collect_h2d_direct_codegen.py \
    --bridge-json "$H2D_DIRECT_OUT/bridge.json" --output-dir "$H2D_DIRECT_OUT/codegen" \
    --cuobjdump "${H2D_CUOBJDUMP:-cuobjdump}" >"$H2D_DIRECT_OUT/codegen-run.log" 2>&1; then
    cat "$H2D_DIRECT_OUT/codegen/codegen-compact.txt"
  else
    H2D_DIRECT_STATUS=$?
    if [[ -f "$H2D_DIRECT_OUT/codegen/codegen-compact.txt" ]]; then cat "$H2D_DIRECT_OUT/codegen/codegen-compact.txt"; fi
    tail -n 30 "$H2D_DIRECT_OUT/codegen-run.log" >&2
    exit "$H2D_DIRECT_STATUS"
  fi
fi
