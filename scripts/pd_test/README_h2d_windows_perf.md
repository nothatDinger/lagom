# Independent copy windows: eight-group perturbation experiment

Physical GPU4 only, NUMA1, scattered host B32/miss100, same source/plans/output,
legacy/control/issue/tail × CTA4/CTA8. Production kernels remain unchanged.

## Reviewed acquisition

Use the existing `work/h2d-windows/check-v1/codegen/codegen.json`, linked to its
16-case check. The runner verifies check source hashes, report hash and four
reviewed binary SHA256 values, then compares actually loaded libraries. No edits
to old performance_ready flags are needed; they were pending-review placeholders.
A different artifact is rejected, not auto-approved.

Review: issue 0770..0860 has no host-data W5 wait, but includes address arithmetic
and 0850 M0x06 non-data dependencies. Tail starts at 0860, after substantial
address arithmetic following the last load 0740; its first store 0890 waits W5,
and its end timer 08c0 has M0x02 read-dependency wait. These are compiler-positioned
windows, not pure issue or PCIe response latency. Both probes group three host
loads on W5; control uses W2/W4/W5. This known difference requires the performance
perturbation gate and limits any resulting attribution.

## Command

Inside one gpuq job allocated physical GPU4, using the existing venv:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_windows_perf.sh \
  "$PWD/work/h2d-windows/check-v1/codegen/codegen.json" \
  "$PWD/work/h2d-windows/perf-v1"
```

New absolute output directory required. No check rerun, NCU, CE or GPU5 run.
The wrapper sets CUDA_VISIBLE_DEVICES=4, CPU/memory node1 and validates GPU UUID.
Before measurement each of eight groups validates eager first/last banks and a
64-bank graph replay, checking payload/padding, guard rows and all interval data.

Eight trials use a four-position Williams ordering twice, with opposite grid
order in the second cycle. Every variant appears twice in every position and
four times with each grid order. Each wave has 64 event-bracketed copies:
64 measured waves, 512 samples/group. Same 250ms warmer, two warmup replays,
validation outside timing and endpoint telemetry as the previous experiment.
Endpoint clocks cannot establish constant state within each wave.

## Gates, defined before running

Keep the previous tolerances: control/legacy ±3%; each probe/control ±5%; each
probe's CTA8/4 ratio relative to legacy's CTA8/4 ±3%. Require both median in band
and at least 7/8 waves in band (ceil(5/6 * 8), preserving the previous required
fraction). Also require positive finite 64 events in each unique scheduled wave,
correct records, matching artifacts and readable P0/Gen5/x16 telemetry.

`success` means run/validation completed. `window_results_valid` requires both
probes' gates. `VALID_BY_PROBE` retains independent issue/tail eligibility, each
including the shared control/legacy and telemetry checks. Gates are diagnostic
screens, not statistical significance tests or proof of unchanged instruction
scheduling. A failed global gate exits nonzero after preserving the full balanced
sequence; do not rerun until a pass or relax thresholds.

Window observations are retained even on failure, labelled as such; only an
eligible probe may inform production hypotheses. Per-KV averages use sum of warp
intervals divided by total KV, not average of different-denominator warp ratios.
CTA8 12-KV/13-KV groups remain separate. Neither sum across warps nor these means
is a kernel critical-path duration. Issue and tail come from separate kernels:
do not add/subtract them into one copy decomposition. Tail end is not HBM store
completion, and a large tail interval cannot identify a specific hardware queue.

## Return

Send `result/windows-compact.txt` in full: eight groups' totals/ratios, separate
perturbation gates, window per-KV/per-warp statistics, record checks and STATE.
Keep `result/windows.json` and `issue-wave*-cta*.json` / `tail-wave*-cta*.json`.
On failure also send the last 30 lines of `run.log`; stop without other runs.
