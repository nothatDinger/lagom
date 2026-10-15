# Round 22: Rx sampling on the same-buffer 64MiB SM/CE comparison

Question: for the current continuous SM implementation on GPU4, does the
lower useful throughput accompany a proportionally lower sampled Rx rate,
or a different sampled Rx per useful byte? Round 21 measured SM 48.62GB/s
versus CE 55.36GB/s. This round measures both quantities together.

## Fixed workload and measurement

Reuse round 21's kernel, exact JIT SHA256, CTA4/1024, shared source/target,
512MiB NUMA1 host pool, seven 64MiB banks, and 12 repeats per chunk graph.
Each chunk has one graph launch and no internal timing Event nodes. Only
physical GPU4 participates, exposed as CUDA0. Portable registration uses GPU4.

Enable the existing Rx-only sampler with a nominal 50ms period. Query duration
and sample spacing are reported separately. A 20ms query duration does not
mean 50Hz sampling. No counter rescaling is applied: raw units are KB/s and
the nominal conversion is raw * 1024 / 1e9 GB/s.

| Global trial | Method | Rx sampler |
|---|---|---|
| 0 | sm64 | OFF |
| 1 | ce64 | OFF |
| 2 | ce64 | ON |
| 3 | sm64 | ON |
| 4 | sm64 | ON |
| 5 | ce64 | ON |
| 6 | ce64 | OFF |
| 7 | sm64 | OFF |

This brackets each method with OFF/ON/ON/OFF and reverses method order in
alternating pairs. Each window uses 2s copy-only warmup, 2.5s measurement,
0.3s edge trimming. ON windows also sample idle traffic before warmup.
Both Rx and useful throughput use the exact same trimmed wall interval.

## Execute

Update as the normal repository owner, outside the gpuq job:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

Inside **one gpuq allocation of physical GPU4**:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
export LAGOM_PY="$PWD/.venv/bin/python3"
bash scripts/pd_test/run_h2d_bulk_compare.sh \
  "$PWD/work/h2d-round22/bulk64-rx" --rx-sampler
```

Use a new output directory. The wrapper handles CUDA_VISIBLE_DEVICES=4,
CPU/memory binding to NUMA1, and run.log. Measurement and warmup total about
40 seconds; allow additional setup time, excluding queueing. No separate
smoke job is needed: the unchanged workload's byte/graph/binding checks run
before measurement in this job. Without the flag, round 21 still runs unchanged.

## Validation and feedback

Required: success/results_valid=True, correctness 14/14, graphs 2/2, windows
8/8, and identity/NUMA/bindings/JIT checks. Retain the existing enqueue,
GPU/wall, chunk stability, Rx coverage/CV and idle-background gates. All four
CE windows must reach 45GB/s; the two ON CE windows must pass the existing
nominal Rx/logical check (0.85 to 1.15). SM has no matching Rx/logical bound.

For each method, ON time versus the mean of its two OFF times and late-OFF
versus early-OFF time must be within 3%. This tighter control is needed to
interpret a roughly 12% rate gap. A quality failure preserves the remaining
balanced windows but suppresses final ratios. A runtime failure saves the
partial report and exits nonzero; no automatic retries.

Return only **bulk64-rx/bulk-compact.txt** (about 20 lines). It includes
OFF/ON/ON/OFF rates, sampler controls, CE checks, four RX records with raw
KB/s and nominal GB/s, query duration/sample spacing, and final ratios.
Keep bulk.json locally. If the compact is absent, return run.log's last 30 lines.

## Interpretation

Compare two ON pairs. Let L be useful SM/CE rate and R be nominal Rx SM/CE
rate; R/L is relative sampled receive traffic per useful byte. Similar R
and L supports lower observed receive rate without a large relative traffic
amplification in this counter. R near one with L lower supports a difference
in receive traffic per useful byte. Intermediate results can contain both.

These aggregate counters guide the next investigation; they do not isolate
request issue latency, completion latency, cache traffic, transaction size,
TLP count, or a physical hardware limit. Do not subtract CE calibration error
or rescale counters to force equality. Prior small-copy Rx is historical
context, not this round's matching baseline.
