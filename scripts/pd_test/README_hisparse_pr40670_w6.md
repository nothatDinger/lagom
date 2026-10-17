# Same-geometry W6 companion and unresolved-location diagnostics

One GPU4 job compares two inputs on the **same W6/K16/H128, scratch13**, same
allocations, initial cache contents, physical mappings, source pool and swap JIT:
8 distinct cold-miss tokens repeated over 96 occurrences, versus 96 distinct
cold misses. B=4 captured rows, three real requests [2,0,1], NUMA1/Portable.
No source-kernel edits, new timing instrumentation, or performance measurements.

The earlier W4/K8/H64/scratch62 companion is useful but changes specialization
and grid, so it cannot isolate scratch-overflow behavior at the failing geometry.
96 misses still fit 128 hot slots. Gather leaving -1 beyond scratch capacity is
an intermediate state: commit has a direct-overflow path and output backfill.
Missing valid outputs remain correctness failures, not tolerated non-admission.

## Run

Inside one gpuq allocation of physical GPU4, existing venv:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_hisparse_pr40670_w6.sh \
  "$PWD/work/hisparse-pr40670/w6-pair-v1"
```

The new files do not overwrite the three locally modified original reference
harness/docs/tests. This entry imports their existing make_case/reset/C4-index
helpers, records the imported harness hash, but uses its own strict content and
state validator. Preserve those uncommitted changes; do not reset them to pull.
The pinned source manifest must match. No cuobjdump is invoked; the prior W6 JIT
can be reused, and a single loaded W6 swap library is required at the end.

Each input has one eager correctness attempt. Passing input proceeds to six
independently reset cold-state Graph replays, then a separate replay with exact
two-kernel launch-shape verification. Failing eager input is recorded without
claiming Graph/profile coverage. Both cases are retained. Overall success is
false if either fails, even when the known missing-token behavior reproduces.

## Return

Return `result/w6-compact.txt` and `result/w6.json`; on runtime failure also return
last 30 lines of run.log. The JSON includes raw final metadata and location maps.
Do not automatically repeat until passing, start benchmarks, loosen the oracle,
or mark this a confirmed PR root cause.

For each real request the diagnostic decodes final committed miss count, scratch
generation, current union-table tokens, compact positions, final index values,
protected hot slots and the token's final hot-cache membership. Missing valid
occurrences are listed. Status distinguishes absent final table entries,
mismatched epochs, UINT32_MAX/no-location, plausible scratch ordinals, and an
already-resolved physical index whose output is still -1. These are **post-commit
observations**, not intermediate-stage measurements: they narrow hypotheses but
do not alone prove where/why a token was lost. No IIO/hardware queue work resumes.

Admitted occurrences must have all 584 bytes correct. Padded rows must be zero;
active invalids must be -1. Unreferenced bytes/guards/padding and permutation of
physical slots are checked independently. Additional byte/state errors are
retained beside unresolved-location failures. Consumer safety remains unverified.
