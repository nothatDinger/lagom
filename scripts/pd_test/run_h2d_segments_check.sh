#!/usr/bin/env bash
# Check and offline codegen only, inside a physical GPU4 gpuq allocation.
set -euo pipefail
if [[ $# -ne 1 || "$1" != /* || -e "$1" ]]; then
  printf 'Usage: LAGOM_PY=/absolute/python bash %s /new/absolute/output\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set existing venv Python}"
[[ "$LAGOM_PY" == /* && -x "$LAGOM_PY" ]] || exit 2
command -v numactl >/dev/null
H2D_SEG_TOOL="$(command -v "${H2D_CUOBJDUMP:-cuobjdump}")"
H2D_SEG_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_SEG_OUT="$1"
cd "$H2D_SEG_REPO"
export PYTHONPATH="$H2D_SEG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_SEG_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
 "$LAGOM_PY" scripts/pd_test/check_h2d_segments.py --output-dir "$H2D_SEG_OUT/check" \
 >"$H2D_SEG_OUT/run.log" 2>&1; then
 cat "$H2D_SEG_OUT/check/check-compact.txt"
else
 H2D_SEG_RC=$?
 [[ ! -f "$H2D_SEG_OUT/check/check-compact.txt" ]] || cat "$H2D_SEG_OUT/check/check-compact.txt"
 tail -n 30 "$H2D_SEG_OUT/run.log" >&2
 exit "$H2D_SEG_RC"
fi
if "$LAGOM_PY" scripts/pd_test/collect_h2d_segments_codegen.py \
 --report "$H2D_SEG_OUT/check/check.json" --output-dir "$H2D_SEG_OUT/codegen" \
 --cuobjdump "$H2D_SEG_TOOL" >"$H2D_SEG_OUT/codegen.log" 2>&1; then
 cat "$H2D_SEG_OUT/codegen/codegen-compact.txt"
else
 H2D_SEG_RC=$?
 [[ ! -f "$H2D_SEG_OUT/codegen/codegen-compact.txt" ]] || cat "$H2D_SEG_OUT/codegen/codegen-compact.txt"
 tail -n 30 "$H2D_SEG_OUT/codegen.log" >&2
 exit "$H2D_SEG_RC"
fi
printf 'STOP: Return compact reports and complete trace/control SASS for boundary review. No performance run.\n'
