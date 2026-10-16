#!/usr/bin/env bash
# Execute inside one gpuq allocation of physical GPU4.
set -euo pipefail
if [[ $# -ne 1 || "$1" != /* || -e "$1" ]]; then
 printf 'Usage: LAGOM_PY=/absolute/python bash %s /new/absolute/output\n' "$0" >&2
 exit 2
fi
: "${LAGOM_PY:?Set existing venv Python}"
[[ "$LAGOM_PY" == /* && -x "$LAGOM_PY" ]] || exit 2
command -v numactl >/dev/null
H2D_BS_TOOL="$(command -v "${H2D_CUOBJDUMP:-cuobjdump}")"
H2D_BS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_BS_OUT="$1"
cd "$H2D_BS_ROOT"
export PYTHONPATH="$H2D_BS_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_BS_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
 "$LAGOM_PY" scripts/pd_test/bench_h2d_blockshape.py --cuobjdump "$H2D_BS_TOOL" \
 --output-dir "$H2D_BS_OUT/result" >"$H2D_BS_OUT/run.log" 2>&1; then
 cat "$H2D_BS_OUT/result/blockshape-compact.txt"
else
 H2D_BS_RC=$?
 [[ ! -f "$H2D_BS_OUT/result/blockshape-compact.txt" ]] || cat "$H2D_BS_OUT/result/blockshape-compact.txt"
 tail -n 30 "$H2D_BS_OUT/run.log" >&2
 exit "$H2D_BS_RC"
fi
