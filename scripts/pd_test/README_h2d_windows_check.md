# Two independent copy timing windows: check and SASS only

The previous five-timestamp trace completed correctly but failed the relative
CTA8/4 perturbation gate: legacy/control were ~1% slower at CTA8, while trace was
~2.6% faster. Do not relax that gate or interpret its residual interval as the
production kernel's precise wait time. The old probe and reports remain intact.

## Change

New diagnostic-only JIT `hisparse_copy_windows` has modes 0/1/2:

- `control`: calls the production copy helper without per-KV instrumentation.
- `issue`: two timestamp reads around the three source loads.
- `tail`: two timestamp reads from after the loads to after the original stores.

Both instrumented modes mirror `transfer_item` lane mapping, tail selection and
store consumers, use explicit `__ldg` to retain read-only load intent, and remove
both the six-address OR predicate and the XOR-based wait-all consumer. Per-KV
state is one interval sum and one monotonicity check, rather than four sums and
five timestamp reads. Per-warp loop start/end and record writeout remain.
Production code and dispatch are unchanged. Old segmented code is unchanged.

This is a candidate reduction in perturbation, not proof of nonperturbation.
Compiler memory clobbers, timers and live registers can still change scheduling.
The new control and instrumented binaries all need fresh SASS review.

`issue` may include source/address instruction scheduling and structural stalls;
it is not PCIe enqueue time. `tail` includes remaining data wait, destination
address work, natural store consumers and scheduling; it is not pure load latency
or HBM store completion. We deliberately give up a pure-wait boundary to avoid
forcing all data ready before the first store. Independent runs' two interval
means MUST NOT be added/subtracted as if they partitioned one invocation.

## Run only check

In one gpuq job allocated **physical GPU4**, reuse the existing venv:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_windows_check.sh \
  "$PWD/work/h2d-windows/check-v1"
```

One new absolute output directory argument. The script sets CUDA_VISIBLE_DEVICES=4,
binds CPU/memory to NUMA1 and verifies GPU4 UUID. It runs four variants × CTA4/8 ×
host/HBM = 16 correctness cases. Each has eager and six mutable graph checks,
including changed counts/real/plans/source, payload/padding and record guards.
The records use 15 columns but only one interval is active: column 6 interval_ns,
column 10 issue_window_ns OR column 12 tail_window_ns; reserved columns 9/11/13
must be zero. Mode-specific validation rejects mixed or incorrectly labelled
intervals. Empty warps have zero interval, all monotonicity flags must be zero.

Then existing cuobjdump extracts all four binaries offline. No NCU, profiler,
performance run, threshold modification, or main entry is invoked. Stops after
extraction; `performance_ready=False` is intentional.

## SASS review before performance

Return:

1. `check/check-compact.txt` and `codegen/codegen-compact.txt`.
2. Complete `codegen/issue.sass.txt`, `tail.sass.txt`, `control.sass.txt`.
3. If failed: compact and last 30 lines of run.log or codegen.log.

We must check:

- Three scalar `LDG.E.64.CONSTANT` and three copy `STG.E.64`, no vector/cache
  policy change; resources/no local or stack spills.
- Two per-KV timers plus loop start/end in each instrumented binary, zero timers
  in control. Record stores after the loop-end timer.
- Issue: start before first load, end after last load, no load-result barrier
  wait before the end timer. Identify address instructions included rather than
  claiming address preparation is excluded.
- Tail: start after last load and before its first data-dependent wait/consumer;
  end after the three stores. Decode the full wait/read/write barrier fields,
  including address-register reuse, and inspect any branch paths.
- Natural per-store data dependencies, without an inserted XOR or forced
  wait-all checkpoint before the first store. Compare actual ordering against
  control, do not rely on source order.

If these checks pass, the next phase will compare legacy/control/issue/tail ×
CTA4/8, retaining absolute and relative-grid perturbation gates. This commit
only provides check/codegen; no claim of fixed GPU behavior until that phase.
