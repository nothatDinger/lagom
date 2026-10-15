#!/usr/bin/env bash
# Run INSIDE a gpuq allocation of physical GPU4. Never modifies device settings.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ( "$1" != check && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/python bash %s check /new/output\n       ... main /new/output /completed/review.json\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set LAGOM_PY to the existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" || "$2" != /* || -e "$2" ]]; then
  printf 'Require executable absolute Python and NEW absolute output directory\n' >&2
  exit 2
fi
if [[ ( "$1" == check && $# -ne 2 ) || ( "$1" == main && $# -ne 3 ) ]]; then
  printf 'check has no review argument; main requires completed review JSON\n' >&2
  exit 2
fi
command -v numactl >/dev/null
H2D_LAT_TOOL="${H2D_CUOBJDUMP:-cuobjdump}"
if [[ "$1" == check ]]; then H2D_LAT_TOOL="$(command -v "$H2D_LAT_TOOL")"; fi
H2D_LAT_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_LAT_OUT="$2"
H2D_LAT_MODE="$1"
H2D_LAT_ARGS=(--mode "$1" --output-dir "$2/$1")
if [[ "$1" == main ]]; then
  [[ "$3" == /* && -f "$3" ]] || { printf 'Require absolute review JSON\n' >&2; exit 2; }
  H2D_LAT_ARGS+=(--review "$3")
fi
cd "$H2D_LAT_REPO"
export PYTHONPATH="$H2D_LAT_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_LAT_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
 "$LAGOM_PY" scripts/pd_test/bench_h2d_latency.py "${H2D_LAT_ARGS[@]}" \
 >"$H2D_LAT_OUT/run.log" 2>&1; then
  cat "$H2D_LAT_OUT/$H2D_LAT_MODE/"*compact.txt
else
  H2D_LAT_RC=$?
  for H2D_LAT_REPORT in "$H2D_LAT_OUT/$H2D_LAT_MODE/"*compact.txt; do
    [[ ! -f "$H2D_LAT_REPORT" ]] || cat "$H2D_LAT_REPORT"
  done
  tail -n 30 "$H2D_LAT_OUT/run.log" >&2
  exit "$H2D_LAT_RC"
fi
if [[ "$H2D_LAT_MODE" == check ]]; then
  if "$LAGOM_PY" scripts/pd_test/collect_h2d_latency_codegen.py \
    --report "$H2D_LAT_OUT/check/latency.json" --output-dir "$H2D_LAT_OUT/codegen" \
    --cuobjdump "$H2D_LAT_TOOL" >"$H2D_LAT_OUT/codegen.log" 2>&1; then
    cat "$H2D_LAT_OUT/codegen/codegen-compact.txt"
  else
    H2D_LAT_RC=$?
    [[ ! -f "$H2D_LAT_OUT/codegen/codegen-compact.txt" ]] || cat "$H2D_LAT_OUT/codegen/codegen-compact.txt"
    tail -n 30 "$H2D_LAT_OUT/codegen.log" >&2
    exit "$H2D_LAT_RC"
  fi
  printf 'Return check/codegen compact reports. Stop here for timer-boundary review; do not run main yet.\n'
fi
