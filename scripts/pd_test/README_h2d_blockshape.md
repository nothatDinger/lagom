# Equal-warp block-shape throughput screen

Question: does grouping the same 128 warps into eight 512-thread CTAs improve
host copy throughput relative to four 1024-thread CTAs? This is not a claim that
eight CTAs occupy eight distinct SMs. Actual SM placement is not measured here.
A positive result would motivate placement measurement; a negative result alone
does not exclude local SM limits without knowing placement.

## Three groups, one GPU4 job

| label | CTAs × threads | warps | KV/warp | request visits |
|---|---|---|---|---|
| c4_t1024 | 4 × 1024 | 128 | 25 | 4096 |
| c8_t512 | 8 × 512 | 128 | 25 | 4096 |
| c8_t1024 | 8 × 1024 | 256 | 12–13 | 8192 |

All use B32/miss100, fixed scattered 512MiB host allocation on NUMA1, Portable
registration, 64 disjoint source banks, identical source plans, destination plan,
counts, target and legacy cache policy. No kernel edits, tail instrumentation,
NCU, Rx, CE pressure, or GPU5. Changing block size creates a separate production
JIT specialization; it can change instruction scheduling and resource usage.

Before timing each shape checks first/last banks eagerly and final graph output,
including untouched/padding bytes. Existing cuobjdump extracts both production
JITs. The runner requires complete control words/resources, no spill, and the
same static scalar load/store counts. A mismatch stops timing and preserves
SASS. Matching counts do NOT establish identical scheduling or occupancy.

Then six permutations of the three-group order, 64 copies/wave, give 18 measured
waves and 384 observations/group. Each has the same 250ms compute warmer and two
graph warmups. Every wave validates output outside timing. Endpoint telemetry is
recorded; it does not establish fixed clocks during waves. Reports contain all
ratios without declaring statistical significance or hardware attribution.

## Execute

Inside a gpuq allocation of physical GPU4:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_blockshape.sh \
  "$PWD/work/h2d-blockshape/main-v1"
```

One new absolute output directory. Uses existing Python and cuobjdump. No manual
edits of old reports or additional check job required; correctness and static
shape gates run before timings in the same job.

Return `result/blockshape-compact.txt`, plus complete
`result/codegen/t512.sass.txt` and `t1024.sass.txt`. Keep blockshape.json locally.
If failed, return compact plus last 30 lines of run.log; do not change gates.

Interpretation: first compare c8_t512/c4_t1024, keeping total warp count and KV
ownership fixed. Check all paired wave ratios and STATE, then inspect codegen.
Only if useful, add a placement observation afterward. Do not infer SM count from
CTA count, or label a throughput change as proof of a particular hardware queue.
