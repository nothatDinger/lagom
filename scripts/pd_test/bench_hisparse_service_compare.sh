#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$3" =~ ^(smoke|main)$ && "$2" =~ ^[a-zA-Z0-9._-]+$ ]] || {
 echo "Usage: LAGOM_PY=/python SHAREGPT_PATH=/data.json bash $0 /out unique-tag smoke|main" >&2; exit 2;
}
: "${LAGOM_PY:?Set Python}" "${SHAREGPT_PATH:?Set local ShareGPT}"
[[ -x "$LAGOM_PY" && -f "$SHAREGPT_PATH" ]] || exit 2
H2D_REAL_OUT=$1 H2D_REAL_TAG=$2 H2D_REAL_MODE=$3
H2D_REAL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$H2D_REAL_ROOT"
export PYTHONPATH="$H2D_REAL_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$H2D_REAL_OUT"
[[ ! -e "$H2D_REAL_OUT/$H2D_REAL_TAG.jsonl" ]] || exit 2
H2D_REAL_N=64 H2D_REAL_C=32 H2D_REAL_O=4096
if [[ "$H2D_REAL_MODE" == smoke ]]; then H2D_REAL_N=2 H2D_REAL_C=1 H2D_REAL_O=128; fi
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
 "$LAGOM_PY" -m sglang.benchmark.serving --backend sglang --tokenize-prompt \
 --host 127.0.0.1 --port 30002 --dataset-name random --dataset-path "$SHAREGPT_PATH" \
 --random-input-len 16384 --random-output-len "$H2D_REAL_O" --random-range-ratio 1 \
 --num-prompts "$H2D_REAL_N" --max-concurrency "$H2D_REAL_C" --warmup-requests 1 --seed 0 \
 --temperature 0 --pd-separated --output-details --output-file "$H2D_REAL_OUT/$H2D_REAL_TAG.jsonl" \
 >"$H2D_REAL_OUT/$H2D_REAL_TAG.log" 2>&1
"$LAGOM_PY" - "$H2D_REAL_OUT/$H2D_REAL_TAG.jsonl" "$H2D_REAL_N" "$H2D_REAL_C" "$H2D_REAL_O" <<'PY'
import json,sys,math
from pathlib import Path
p=Path(sys.argv[1]); n,c,o=map(int,sys.argv[2:]); rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
if len(rows)!=1: raise ValueError('Expected exactly one result')
r=rows[0]; expected=dict(completed=n,total_input_tokens=n*16384,total_output_tokens=n*o,max_concurrency=c)
errors={k:dict(expected=v,actual=r.get(k)) for k,v in expected.items() if r.get(k)!=v}
if any(r.get('errors', [])): errors['request_errors']=r['errors']
for field in ['mean_tpot_ms','median_tpot_ms','p95_tpot_ms','output_throughput','mean_ttft_ms']:
 if not isinstance(r.get(field),(int,float)) or not math.isfinite(r[field]) or r[field]<=0: errors[field]='invalid metric'
report=dict(valid=not errors,expected=expected,errors=errors,metrics={k:r.get(k) for k in ['mean_tpot_ms','median_tpot_ms','p95_tpot_ms','output_throughput','mean_ttft_ms','duration']})
p.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
p.with_suffix('.compact.txt').write_text(json.dumps(report,ensure_ascii=False)+'\n')
print(json.dumps(report,ensure_ascii=False))
if errors: raise SystemExit(1)
PY
