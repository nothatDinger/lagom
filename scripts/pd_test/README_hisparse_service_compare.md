# DSPARK TP4 P/D: PR fused versus branch split swap-in

This is the requested return to a real service workload. The preceding W6/K16
synthetic result is not reused as a TPOT estimate. There is no local GPU/model,
and the service integration has not been executed in this development environment.

## Fixed workload and two implementations

- Single host, prefill physical0-3 TP4, decode physical4-7 TP4; eight GPUs total.
- Local DeepSeek-V4-Flash-0731 model, local ShareGPT seed0 as the prompt source.
  Exact tokenized input16384/output4096, 64 requests, maximum concurrency32,
  ignore EOS, temperature0, one warmup excluded. Random dataset mode retains
  the established ShareGPT-backed prompt sampling with exact lengths.
- DSPARK block-size5; read actual verify width from server/probe, don't assume
  it equals block-size. Top-k is requested512; the effective model index_topk
  must also be512. Hot3072 and host/device ratio5 match prior runs.
- force_h2d_each_step=true. This invalidates residency, NOT duplicate selections.
  Actual unique host misses can be less than512*width because steps can select
  the same token and current provisional C4 can belong to writer scratch.
- `split` (default): existing whole-window metadata gather+commit then legacy
  copy_cache_planned_mla, CTA4/1024. `pr_fused`: pinned PR plus previously
  validated CAS-retry correction, gather/commit with embedded transfer.
  The PR bodies/launches are unchanged. No independent copy launches in this mode.
- Existing GPU masks, model settings and Mooncake NIC map follow round1.
  CPU/memory affinity is NUMA0 for prefill and NUMA1 for decode in BOTH groups.

## Integration boundary and correctness gate

The PR experiment is opt-in via SGLANG_HISPARSE_VERIFY_IMPL=pr_fused. It requires
CUDA DSv4 speculative verify, real IO, force-H2D and no named prefetcher. Normal
serving default remains split. It rejects requests whose compressed prefix is
not longer than hot (short-context positional reads are not covered).

Use each request's ALREADY ALLOCATED hot+64-slot pool: keep canonical and the
max-width E placeholder slots fixed; remaining tail slots form PR KV scratch.
For width6/hot3072 this is active3075, scratch61. PR hot/scratch locations may
rotate PER LAYER, but retain the same per-request physical allocation set;
request_finished frees that original set once. Canonical stays fixed for the
existing DSPARK accepted-C4 writer. Independent coordinator-owned DSPARK writer
scratch/selection/commit code is untouched. This differs from the earlier
synthetic scratch13 geometry; it is not a pure change of batch or top-k.

First sync branch while preserving the user's dirty reference files. On GPU4,
run the new rotation/content regression and the existing window-hash tests:

```bash
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=4 "$LAGOM_PY" -m pytest -q \
 test/registered/kernels/ops/kvcache/test_hisparse_window_hash.py \
 test/registered/unit/test_hisparse_service_pipeline.py \
 test/registered/unit/managers/test_hisparse_dspark.py
```

Require actual CUDA passes, not skipped CUDA tests. The new PR test checks full
KV bytes and per-request allocation permutation over eight forced-cold windows,
with alternating input tokens and retained rotated mappings; canonical/E must
stay fixed. Existing tests cover local hash, invalid/padded rows and rollback.
Stop and return errors on failure. This gate does not prove all model-level
consumer semantics; the following service smoke checks are also required.

## Start each service group

Reserve eight GPUs in one gpuq job/session. Launcher runs in foreground and owns
only its own process groups; Ctrl-C stops them. It refuses occupied service ports,
never killall's existing services. Use a second session in the same allocation
for client commands. All output directories must be new.

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/
export SHAREGPT_PATH=/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json
export H2D_SERVICE_OUT="$PWD/work/hisparse-service-v1"
# Run one at a time; retain logs. Start with split; then repeat for pr_fused.
bash scripts/pd_test/run_hisparse_service_compare.sh split off "$H2D_SERVICE_OUT/split-off"
```

Wait for readiness of all endpoints. Server's short dummy HTTP warmup is disabled
in BOTH groups because PR integration currently admits only long requests.
The client performs an explicit 16k warmup instead. Check decode logs for the
actual backend, top-k, hot size and verify width; defaults/graph-captured B are
not a substitute for real batch. Disable other named/shared prefetch, skip-IO,
load-policy overrides identically in both groups (launcher handles this).

In second session, from repo with same exported Python/dataset/output variables:

```bash
bash scripts/pd_test/bench_hisparse_service_compare.sh "$H2D_SERVICE_OUT/bench" split-smoke smoke
# Only after smoke's valid=true and no backend/assert/CUDA errors:
bash scripts/pd_test/bench_hisparse_service_compare.sh "$H2D_SERVICE_OUT/bench" split-1 main
bash scripts/pd_test/bench_hisparse_service_compare.sh "$H2D_SERVICE_OUT/bench" split-2 main
```

Stop launcher with Ctrl-C, start pr_fused/off in a NEW directory, and run
pr-smoke, pr-1, pr-2 in the same way. Smoke uses two long requests, concurrency1,
128 output. Compare saved generated_texts/errors and accept-length logs between
smokes: unresolved locations, corruption, CUDA errors or unexpected semantic
output changes block performance conclusions. Any generated-text divergence
needs review; token equality is not itself a full cache oracle.

Unprofiled main runs validate all64 successes and exact total1048576 input /
262144 output tokens, recording mean/median/p95 TPOT, throughput and TTFT.
Keep raw JSONL (including output details) locally. If a material improvement is
seen, repeat with reversed implementation order before calling it a gain.

## Separate latency profile, for EACH implementation

Restart the chosen backend with `on` instead of `off`, new directory (e.g.
`split-on`). Do not include profiled TPOT in the performance comparison.
Probe capacity16384, maxbatch1024, no per-operation CPU sync/event read in service.
Start bounded collector in second session; it waits for two NEW B32 decode logs:

```bash
"$LAGOM_PY" scripts/pd_test/collect_hisparse_service_profile.py \
 --decode-log "$H2D_SERVICE_OUT/split-on/decode.log" \
 --output-dir "$H2D_SERVICE_OUT/split-profile" --backend split &
H2D_COLLECTOR_PID=$!
bash scripts/pd_test/bench_hisparse_service_compare.sh "$H2D_SERVICE_OUT/profile-load" split main
wait "$H2D_COLLECTOR_PID"
```

Repeat for `pr_fused` with pr-on/pr-profile/pr-fused labels. No profiler or NCU
upgrade needed; existing Torch/CUPTI capture exports individual graph nodes.
The collector expects exactly four rank artifacts with matching capture/PID.
It reports gather/commit/copy activity durations and complete pipeline span
(first gather start to last commit/copy end, including gaps and any PDL overlap;
force-residency invalidation and hash-state initialization before gather are
outside this span, but remain included in unprofiled service TPOT).
PR copy column is N/A, not zero and not an inferred standalone copy duration.
Do NOT sum gather+commit blindly when PDL overlaps them.

Actual unique misses are joined by serialized recorder order and GPU stream.
PR counts are read from its existing counters, with a profile-only batch-order
index_select after commit; RecordMissPlan remains false. Recorder/count-extract
activities lie outside pipeline span, but can perturb the workload. Compare
kernel numbers only in similar B/W/actual-miss buckets; don't sum TP ranks.

## Feedback / interpretation

Return these compact artifacts first (raw JSON/traces stay on machine):

1. CUDA/CPU gate summary and split/pr-smoke validation plus any output discrepancy.
2. Four bench *.compact.txt files (split-1/2, pr-1/2), startup config.txt and the
   decode lines giving backend/effective top-k/hot/verify-width and accept stats.
3. split-profile/service-pipeline-compact.txt and pr-profile counterpart.
   They list every rank, B/W, actual miss distribution, each C4 layer's mean
   gather/commit/copy/whole-pipeline latency and grouped mean/median.
4. Failure: failing log tail30 and compact/error, then stop; don't lower gates.

Desired comparison table PER RANK and similar actual load:

| backend | real B / W / unique misses | gather us | commit us | copy us | whole swap us | unprofiled TPOT |
| split | observed | activity | activity | activity | span | separate run |
| pr_fused | observed | includes transfer | includes transfer | N/A | span | separate run |

Pipeline spans are synchronous swap-in latency, not physical PCIe instantaneous
bandwidth. A complete repeated layer cycle additionally reports synchronous
pipeline time / verify-to-verify cadence. That is a profiled cadence fraction,
NOT the percentage of TPOT caused by H2D: TPOT depends on accepted output tokens,
draft/attention/communication, cross-rank waits and scheduler work. Don't divide
sums from all TP ranks by a request's TPOT. Full service correctness, matched load,
critical-path shortening and reproducible unprofiled TPOT must agree.

Development validation limitation: this machine has no GPU/model and lacks
transformers needed to import the entire coordinator test suite. Lightweight
backend dispatch and trace analyzer CPU tests run locally; the CUDA and full
manager tests above must pass on the existing GPU environment before service.
