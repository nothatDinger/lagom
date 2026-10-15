# Round 21: same-size 64MiB continuous SM and CE, physical GPU4 only

This closes the size mismatch between the previous 1.87MB continuous SM result
(GPU4 about 46GB/s) and 64MiB bulk CE (about 55GB/s). It tests this continuous SM
implementation, not an absolute hardware limit or production gather speedup.

## Fixed comparison

| Method | Bytes per copy | Kernel / API |
|---|---|---|
| sm64 | 64MiB | Existing scalar `hisparse_linear_copy_kernel`, CTA4/1024 |
| ce64 | 64MiB | Existing `Tensor.copy_(non_blocking=True)` CE path |

Both methods read the same seven contiguous views of one 512MiB registered host
allocation and write the same CUDA output view and its two guard regions.
The pool is fully mapped on NUMA1 (`bind:1`, reported4KiB pages); CPU and memory
are bound to node1. CUDA0 is physical GPU4 (`98:00.0`), registered with Portable
flags=1 on that GPU. No second GPU is used or needed.

Both chunk graphs contain 12 repeats of the same 7-bank sequence (84 copies),
with one launch per chunk and no per-copy timing Event nodes. Method order is
SM,CE,CE,SM. Each window has 2s of copy-only warmup, 2.5s measurement and 0.3s
edge trimming. No compute preheat, Rx sampling, NCU, CTA sweep or CUDA change.

Only the Python `ContiguousCase` accepts a new optional `nbytes`; its default
remains 1,868,800B so existing experiments retain their sizes. The loaded linear
JIT must match the exact artifact hash from the successful round 20 run.

## Execute

As the normal repository owner, update the existing branch:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

Then, **inside one gpuq allocation of physical GPU4**, run:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
export LAGOM_PY="$PWD/.venv/bin/python3"
bash scripts/pd_test/run_h2d_bulk_compare.sh "$PWD/work/h2d-round21/bulk64"
```

The wrapper sets CUDA_VISIBLE_DEVICES=4 and numactl itself. Use a fresh output
directory. Do not submit separate jobs for the two methods. Runtime should be
around a minute excluding queue and existing JIT cache setup.

## Checks and feedback

Before timing, both methods check every payload and guard byte for all 7 banks
(14 eager checks), then validate two reset/replayed sustained graphs. Source,
target, guard, byte/bank counts, alignment and equal chunk work are checked.
A failed preflight stops before measurement. Timing retains all four balanced
windows even when quality fails; no final ratios are printed in that case.

Required: success=True, correctness 14/14, graphs 2/2, identity/NUMA/bindings/JIT
all True, timing 4/4, original submission and steady-state limits, <=10% method
repeat drift, and both CE windows >=45GB/s. No gate requires SM to achieve a
particular rate; its measured gap is the question.

Return only **bulk64/bulk-compact.txt** (about 13 lines). It includes both methods'
two wall/GPU rates, quality and clock/state endpoints, and SM/CE time/rate ratios.
On failure, return the same compact; if absent, provide run.log's last 30 lines.
Keep bulk.json on the machine. Local CPU checks do not replace this native
64MiB byte/graph verification.

If SM approaches CE, that supports size-dependent efficiency in this implementation;
it does not establish that all size effects are launch overhead. If SM remains
slower, this baseline retains a gap at 64MiB; it still cannot prove a hardware
ceiling. The old small-size result is historical context, not a same-run size sweep.
