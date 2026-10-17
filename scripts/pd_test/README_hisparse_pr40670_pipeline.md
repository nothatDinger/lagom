# Full verify-window swap: PR retry candidate versus local planner + copy

## Purpose and scope

Return to the serving-path question: how much time does the complete verify
swap-in take with each implementation? This is a synthetic correctness-gated
comparison, not a TPOT result or proof of a hardware bandwidth limit.

- `pr_retry_fixed`: pinned PR #40670 plus the single CAS-retry accounting fix
  validated by the preceding reproducer. Fused gather + commit, including their
  own KV transfers; gather `[B,2,1] x 64`, commit `[B,1,1] x 512`.
- `local_plan_copy`: current local whole-window metadata gather + commit, then
  production `copy_cache_planned_mla`, explicit legacy, CTA4/1024. Same gather
  and commit launch dimensions plus `[4,1,1] x 1024` copy.

Do not label the candidate as unmodified PR. No original-PR failure is accepted
as a performance baseline. Serving dispatch is unchanged. Both paths now carry
that same lost-CAS retry-accounting correction; no copy-kernel configuration or
KV transfer instruction is changed.

Use physical GPU4 only, NUMA1, W6/K16, hot128, PR tail16/scratch13, real batch4
and batch32, unique cold misses8 and96 per request. The two implementations share
one Portable-registered host allocation for each case, exact input token lists,
initial hot contents, host-index mapping and physical destination pool geometry.
Each owns independent output/state; final physical locations may differ. PR
retains its scratch rotation; local retains fixed hot+canonical slots and no KV
scratch. Those differences, including possible additional PR KV transfers, are
part of the implementation comparison, not isolated copy-code effects.

The token distribution is inherited from the W6 reproducer, not a production
DSPARK trace. All B requests are real, with request order rotated. Earlier checks
cover padded/invalid rows; this experiment measures cold valid work only.

## Run

Preserve existing dirty reference harness files; their hash is recorded. From
repository root, after fast-forwarding the branch, submit the following inside
one gpuq allocation for GPU4:

```bash
LAGOM_PY=/absolute/path/to/existing/venv/bin/python \
  bash scripts/pd_test/run_hisparse_pr40670_pipeline.sh \
  "$PWD/work/hisparse-pr40670/pipeline-v3"
```

Use a new absolute output directory. No NCU, cuobjdump, root changes, NVML Rx
sampling, CE baselines or software installation is needed.

The wrapper runs separate `check` then `main` processes. Any check failure stops
before main. Both phases verify each method with eager, six cold-reset Graph
replays, and one profiled replay. Verify all 584 KV bytes, consistent duplicate
resolution, unique output counts, unreferenced bytes/guards, PR physical-slot
permutation, local fixed slots and miss counts. Profiling requires exactly two
PR kernels or three local kernels with the expected order, grid, block and GPU.
Both check and main also check the timed reset graph and read a positive finite
event interval before sampling. A new correctness failure
is recorded and stops the job; don't relax its oracle to get timing.

## Timing definition

Every sample is a cold logical cache state, not repeated warm hits. An initialized
GPU snapshot restores output, token/slot tables, hash/CLOCK state and local plan
buffers. A CUDA graph places these D2D resets BEFORE a start event, then the
complete pipeline, then an end event. The resets and host synchronization are
outside the measured event interval; intermediate GPU kernels and their gaps
are included. No host pinning, state initialization, correctness checking or
profiler runs occur inside timed event intervals.

Six waves alternate PR/local then local/PR (three of each order), 32 samples per
method per wave, 192 samples/method/case. Eight warm-up replays also restore cold
state. Event timing reflects this controlled snapshot/caching environment, not
an actual request arrival, network transfer, or attention overlap. Reset graph
nodes and event nodes may perturb tiny workloads; use paired wave consistency
and the separately profiled single-replay activity durations as context.
The main metric is total microseconds, not H2D GB/s: different scratch behavior
can generate different actual byte traffic despite identical logical requests.
Power/clock telemetry is sampled only at wave boundaries and does not prove
constant state during a wave. Retain it to identify obvious drift.

## Feedback and decision

Send:
1. `check/result/pipeline-compact.txt` (one-line gate is sufficient if all pass).
2. `main/result/pipeline-compact.txt` in full: four RESULT rows contain total
   means, six wave means and six local/PR ratios; CHECK rows contain each
   profiled kernel duration and launch shape. STATE and code/JIT provenance are
   included. Ratio >1 means local is slower.
3. On failure, the failing phase's compact plus the last 30 lines of `check.log`
   or `main.log`. Keep JSON/traces locally; don't paste them unless requested.

First check all gates. Then compare full-pipeline total latency, especially B32
96-miss; inspect per-kernel activity for where a difference occurs, without
subtracting different profiled and unprofiled runs. Small/no difference means
moving copy between these implementations is not yet a promising optimization.
A consistent material difference motivates a subsequent service integration and
TPOT test, with correctness and writer/scratch safety checked there. Neither
outcome alone predicts DSPARK high-concurrency TPOT.


## v1 failure and v2 correction

The first GPU check passed B4/8 for both paths and B4/96 for the PR candidate,
then stopped at the local B4/96 eager check with unresolved valid KV. No main
performance result was produced. Inspection found the local adaptation still
had the original CAS-loser `continue` consuming a probe allowance. The PR-only
correction had not been propagated into this inherited local code.

V2 adds the same `--attempt` immediately before that retry, retaining every other
local planner/copy setting. The local source hash and content-addressed JIT
artifact change; there is no reason to delete shared caches. Keep the v1 report
as the uncorrected-local observation. V2 must pass the strict check before its
main timing runs; this GPU rerun, not source inspection alone, will determine
whether the correction resolves this failure.

A named failure now records active batch/miss/method/stage, unresolved token and
location details, local miss counts, and final union-table capacity/epoch/count
and missing-token sets. Details are post-commit observations, not a timeline of
the race. These appear as FAILURE in compact and failure_diagnostic in JSON.
A CUDA regression adds the full 96-slot cold-union case in eager and six Graph
replays; CPU tests do not substitute for executing that regression on a GPU.


## v3: external event nodes and check/main parity

V2 main passed the first case's eager/Graph/profile checks for both methods,
then failed at `end.synchronize()` with CUDA invalid argument. The harness had
used default captured events, which are graph dependencies instead of externally
observable event-record nodes. This is a timing-harness defect, not evidence of
a new KV failure. No valid performance result was produced.

V3 uses `torch.cuda.Event(enable_timing=True, external=True)`, matching the
already-used capture_wave helper. Both check and main now replay the exact timed
cold-reset graph, synchronize its end event, read a finite positive duration,
and validate final KV contents. CHECK includes timing_probe_us; it is an API
smoke check, not a performance result. Six-wave main timing uses the same reader
and rejects zero/nonfinite durations. Reset remains outside the event interval.
A CUDA regression captures two separate toy pipelines, then alternates repeated
replays and validates host-readable timestamps and reset contents. CUDA checks
still require real hardware; CPU success alone does not validate this API.
