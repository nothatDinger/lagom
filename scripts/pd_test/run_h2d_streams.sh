#!/usr/bin/env bash
# Run inside a physical GPU4 gpuq allocation.
set -euo pipefail
if [[ ( $# -ne 2 && $# -ne 3 ) || ( "$1" != smoke && "$1" != main ) ]]; then
  printf 'Usage: LAGOM_PY=/absolute/python bash %s {smoke|main} /new/output [host|hbm]\n' "$0" >&2
  exit 2
fi
: "${LAGOM_PY:?Set existing lagom venv Python}"
if [[ "$LAGOM_PY" != /* || ! -x "$LAGOM_PY" || "$2" != /* || -e "$2" ]]; then
  printf 'Require executable absolute Python and NEW absolute output directory\n' >&2
  exit 2
fi
H2D_STREAM_SOURCE="${3:-host}"
if [[ "$H2D_STREAM_SOURCE" != host && "$H2D_STREAM_SOURCE" != hbm ]]; then
  printf 'Source must be host or hbm\n' >&2
  exit 2
fi
command -v numactl >/dev/null
H2D_STREAM_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
H2D_STREAM_OUT="$2"
cd "$H2D_STREAM_REPO"
export PYTHONPATH="$H2D_STREAM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_STREAM_OUT"
if CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_streams.py --mode "$1" --source "$H2D_STREAM_SOURCE" \
  --output-dir "$H2D_STREAM_OUT/result" >"$H2D_STREAM_OUT/run.log" 2>&1; then
  cat "$H2D_STREAM_OUT/result/streams-compact.txt"
else
  H2D_STREAM_RC=$?
  [[ ! -f "$H2D_STREAM_OUT/result/streams-compact.txt" ]] || cat "$H2D_STREAM_OUT/result/streams-compact.txt"
  tail -n 30 "$H2D_STREAM_OUT/run.log" >&2
  exit "$H2D_STREAM_RC"
fi
