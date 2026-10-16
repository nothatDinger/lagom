#!/usr/bin/env bash
# Inside a GPU4 gpuq allocation. Uses reviewed windows check-v1, does not rerun check.
set -euo pipefail
if [[ $# -ne 2 || "$1" != /* || ! -f "$1" || "$2" != /* || -e "$2" ]]; then
  printf 'Usage: LAGOM_PY=/absolute/python bash %s /absolute/codegen.json /new/absolute/output\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set existing venv Python}"
[[ "$LAGOM_PY" == /* && -x "$LAGOM_PY" ]] || exit 2
command -v numactl >/dev/null
H2D_SEG_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_SEG_CODEGEN="$1"
H2D_SEG_OUT="$2"
cd "$H2D_SEG_REPO"
export PYTHONPATH="$H2D_SEG_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_SEG_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
 "$LAGOM_PY" scripts/pd_test/bench_h2d_windows.py \
 --codegen "$H2D_SEG_CODEGEN" --output-dir "$H2D_SEG_OUT/result" \
 >"$H2D_SEG_OUT/run.log" 2>&1; then
 cat "$H2D_SEG_OUT/result/windows-compact.txt"
else
 H2D_SEG_RC=$?
 [[ ! -f "$H2D_SEG_OUT/result/windows-compact.txt" ]] || cat "$H2D_SEG_OUT/result/windows-compact.txt"
 tail -n 30 "$H2D_SEG_OUT/run.log" >&2
 exit "$H2D_SEG_RC"
fi
