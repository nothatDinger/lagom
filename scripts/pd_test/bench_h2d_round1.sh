#!/usr/bin/env bash
set -euo pipefail

# Usage: LAGOM_PY=/path/to/venv/bin/python3 SHAREGPT_PATH=/path/to/dataset.json \
#   bash scripts/pd_test/bench_h2d_round1.sh /path/to/output baseline-1
# Run from the repository root. Profile separately on the decode endpoint.
H2D_OUT=${1:?Pass an output directory}
H2D_TAG=${2:?Pass a unique run tag}
LAGOM_PY=${LAGOM_PY:-python3}
SHAREGPT_PATH=${SHAREGPT_PATH:?Set SHAREGPT_PATH}
test -f "$SHAREGPT_PATH"
mkdir -p "$H2D_OUT"
if [[ -e "$H2D_OUT/$H2D_TAG.jsonl" ]]; then
    printf 'Result already exists; choose a new run tag: %s\n' "$H2D_OUT/$H2D_TAG.jsonl" >&2
    exit 1
fi
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$LAGOM_PY" -m sglang.benchmark.serving \
    --backend sglang --tokenize-prompt --host 127.0.0.1 --port 30002 \
    --dataset-name random --dataset-path "$SHAREGPT_PATH" \
    --random-input-len 16384 --random-output-len 4096 --random-range-ratio 1 \
    --num-prompts 64 --max-concurrency 32 --warmup-requests 1 --seed 0 \
    --pd-separated --output-file "$H2D_OUT/$H2D_TAG.jsonl" \
    >"$H2D_OUT/$H2D_TAG.log" 2>&1

"$LAGOM_PY" - "$H2D_OUT/$H2D_TAG.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads([line for line in path.read_text().splitlines() if line.strip()][-1])
expected = dict(completed=64, total_input_tokens=64*16384, total_output_tokens=64*4096)
differences = {key: dict(expected=value, actual=result.get(key))
               for key, value in expected.items() if result.get(key) != value}
validation = dict(passed=not differences, expected=expected, differences=differences)
path.with_suffix('.validation.json').write_text(json.dumps(validation, indent=2) + '\n')
if differences:
    raise SystemExit('Workload validation failed: ' + json.dumps(differences))
print('Workload validation passed: 64 requests, exact 16K input / 4K output')
PY
