#!/usr/bin/env bash
# Run inside one physical-GPU4 gpuq job; check/profile/codegen only.
set -euo pipefail
if [[ $# -ne 1 || "$1" != /* || -e "$1" ]]; then
 printf 'Usage: LAGOM_PY=/absolute/python bash %s /new/absolute/output\n' "$0" >&2
 exit 2
fi
: "${LAGOM_PY:?Set existing venv Python}"
[[ "$LAGOM_PY" == /* && -x "$LAGOM_PY" ]] || exit 2
command -v numactl >/dev/null
H2D_PR_TOOL="$(command -v "${H2D_CUOBJDUMP:-cuobjdump}")"
H2D_PR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_PR_OUT="$1"
cd "$H2D_PR_ROOT"
export PYTHONPATH="$H2D_PR_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_PR_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
 "$LAGOM_PY" scripts/pd_test/check_hisparse_pr40670.py \
 --cuobjdump "$H2D_PR_TOOL" --output-dir "$H2D_PR_OUT/result" \
 >"$H2D_PR_OUT/run.log" 2>&1; then
 cat "$H2D_PR_OUT/result/reference-compact.txt"
else
 H2D_PR_RC=$?
 [[ ! -f "$H2D_PR_OUT/result/reference-compact.txt" ]] || cat "$H2D_PR_OUT/result/reference-compact.txt"
 tail -n 30 "$H2D_PR_OUT/run.log" >&2
 exit "$H2D_PR_RC"
fi
printf 'STOP: review reference correctness and launch/codegen before benchmarks or serving integration.\n'
