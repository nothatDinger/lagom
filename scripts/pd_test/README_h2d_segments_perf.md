# Segmented copy: six-group timing and perturbation validation

Scope: physical GPU4 only, NUMA1 pinned scattered host source, B32/miss100,
legacy/control/trace × CTA4/CTA8. No kernel change, NCU, NVML Rx, CE load,
serving integration, or GPU5 experiment.

## Reviewed prerequisite

The complete v2 trace/control SASS was reviewed in the conversation. The runner
binds to the three reviewed binary SHA256 values in `hisparse_h2d_segments_perf.py`.
It checks the existing codegen report, its linked 12-case correctness report,
unchanged correctness source hashes, and the actual loaded libraries. It does not
change old `performance_ready=False` or `review-template.json`. Those flags were
placeholders pending the review now completed, not failed correctness checks.
A different build is rejected, not silently approved. Keep check-v2 in place.

T0→T1: plan/source/destination preparation. T1→T2: execution of three load
instructions, including possible scheduling or structural backpressure. T2→T3:
remaining data wait plus dependent consumers/scheduling. T3→T4: store instruction
section including checksum operations; not HBM completion. T2 has no W5 wait;
0a90 waits M0x20 before consumers; T3/T4 M0x02 involves address reuse.
Trace changes overlap: destination preparation precedes loads and all data are
consumed before stores. Segment attribution is therefore conditional on timing
perturbation validation, and never proves a particular PCIe queue or lock.

## Execution

Inside **one gpuq job allocated physical GPU4**, using the existing venv:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_segments_perf.sh \
  "$PWD/work/h2d-segments/check-v2/codegen/codegen.json" \
  "$PWD/work/h2d-segments/perf-v1"
```

The second argument must be new. No `main` argument is added to the old check
script. The new runner sets CUDA_VISIBLE_DEVICES=4 and verifies GPU4's UUID.
No second GPU job or new check/codegen run is needed if source/binaries match.

Before timing, all six groups check first/last banks eagerly, then graph replay,
payload/padding and record guards. Each group uses the same 512 MiB allocation,
64 source banks, plans and output. Each wave has 64 event-bracketed invocations;
six balanced permutations of variant order and alternating grid order yield
36 measured waves, 384 observations/group. The existing 250 ms compute warmer
and two graph warmups are used equally; endpoint STATE is retained and does not
prove constant clocks inside a wave. CPU validation and record downloads are
outside timing. Every measured wave rechecks final-bank output and all records.

## Gates and interpretation

The existing round23 conservative perturbation gates are reused, not fitted to
these results: control/legacy within ±3%, trace/control within ±5%, and relative
CTA8/4 ratio (trace vs legacy) within ±3%. Each requires the six-wave median and
at least five of six individual ratios inside the band. Telemetry must be
readable, P0, Gen5 x16. Frequency and throttle readings are reported, not treated
as proof of no perturbation.

`success=True` means execution and validation completed. `segment_results_valid`
also requires the perturbation and telemetry gates. A gate failure exits nonzero
but preserves all balanced observations: do not rerun until one happens to pass.
Raw segment statistics remain labelled `SEGMENT_OBSERVATION` even on a failed
gate, for debugging; they must not be used to attribute production behavior.

Per-KV means use sum of warp segment durations / sum of copied KV. Per-warp means
use sum / number of warps. Neither is a kernel critical-path duration. CTA8
12-KV and 13-KV warps are also reported separately. Summaries and raw records
preserve warp distributions, last-finishing warp and launch envelope. Each record
must have monotonic boundaries, exact four-segment sum, correct KV/visit counts,
valid guards, and copy duration within its loop span.

First check total CTA8/4 behavior and instrumentation perturbation. Only then
compare which segment grows. Total-time agreement is necessary, not proof that
instrumentation perfectly preserves internal overlap. Do not call issue time
PCIe enqueue time, residual time pure link latency, or store time HBM completion.

## Feedback

Return `result/segments-compact.txt` in full. Keep `result/segments.json` and
`trace-wave*-cta*.json` locally; send them if the compact statistics need review.
The compact includes six groups' per-wave/mean times, perturbation ratios,
trace four-segment per-KV/per-warp means, 12/13/25-KV groups, record checks and
per-wave before/after STATE. On failure include the compact and last 30 lines of
`run.log`; stop, do not adjust thresholds or continue other experiments.
