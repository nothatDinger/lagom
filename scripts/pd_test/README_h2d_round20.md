# Round 20: reverse CUDA enumeration of physical GPU4/5

Question: does the host-SM throughput difference follow the physical GPU/path,
or the logical device / first explicit CUDA allocation? This control changes
enumeration and allocation order together; it cannot distinguish those two
process factors from each other, or separate a board from its PCIe path.

## Workload and controls

One gpuq allocation reserves physical GPU4/5 throughout. The wrapper starts two
separate Python processes, in order:

| Process | CUDA_VISIBLE_DEVICES | First explicit CUDA tensor allocation | Portable registration |
|---|---|---|---|
| normal | 4,5 | CUDA0 = physical GPU4 | CUDA1 = physical GPU5 |
| reversed | 5,4 | CUDA0 = physical GPU5 | CUDA0 = physical GPU5 |

Both explicitly allocate on CUDA0, then CUDA1, before allocating/registering
the host pool. The report records whether PyTorch CUDA was already initialized,
allocation order, UUID/BDF identities, registration device, PID and runtime.
These observations do not establish every internal driver initialization event.

Each process uses one shared 512MiB host pool on NUMA1 (`bind:1`, 4KiB pages),
Portable flags=1, CPU/memory binding to node1. Host allocations and virtual
addresses differ between processes; both devices within a process share one pool.

Only two existing methods are timed:

- `linear4`: existing scalar legacy-helper continuous SM copy, 1,868,800B,
  CTA4/1024, 64 banks. Fixed 32 bank-sequence repeats per chunk in both processes.
- `ce_bulk`: existing 64MiB CE copy, 7 banks, fixed 12 repeats per chunk.

No new CUDA kernel or read policy. A per-device capture stream and one graph
launch per chunk reuse the round-16 timing fix. No Rx sampler or compute warmup.
Each window has 2s of copy warmup, 2.5s measurement, 0.3s edge trimming.
Logical device order is ABBA (0,1,1,0); method order reverses on each device's
second visit. Both devices thus receive both method orders in each process.

## Execute

From the repo root, inside the existing gpuq allocation of **physical GPU4/5**,
using the existing venv:

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_device_order.sh "$PWD/work/h2d-round20/device-order"
```

Use a fresh output path. Keep both processes in this one GPU allocation. The
wrapper itself sets CUDA_VISIBLE_DEVICES and numactl; do not wrap it in an
additional Python worker or change its device order externally. This takes a
few minutes excluding queue time and any existing JIT cache load/compilation.

Each process first checks 16 linear eager/graph/guard boundary cases (including
HBM controls), then verifies four sustained graphs and shared pool bindings.
All checks must pass before measurement. A failed acquisition stops the wrapper;
quality failures retain the complete ABBA sequence within that process.

Main timing requires all eight windows, original submission/steady-state quality
limits, <=10% repeat drift, and all four bulk CE windows >=45GB/s. The final
comparison additionally requires matching physical UUID/BDF identities, one
identical loaded linear JIT SHA256, matching source hashes/runtime and distinct
process IDs. Failure suppresses the final CROSS lines.

## Feedback

Return only **`work/h2d-round20/device-order/order-compact.txt`** (about 35 lines).
It contains both mappings, physical-card throughput, CE reference, clock/state
endpoints, validation gates and GPU5/GPU4 time ratios for both processes.

On failure, return the failed process's `normal/order-compact.txt` or
`reversed/order-compact.txt`; if missing, return the last 30 lines of its run.log.
If both workers passed but summary failed, return `summary.log`.
Keep raw JSON locally. The offline summary can be rerun without GPU work:

```bash
PYTHONPATH=python "$LAGOM_PY" scripts/pd_test/bench_h2d_device_order.py summarize \
  --normal work/h2d-round20/device-order/normal/order.json \
  --reversed work/h2d-round20/device-order/reversed/order.json \
  --output-dir work/h2d-round20/device-order-resummary
```

If physical GPU4 stays faster under reversal, the gap follows the physical
device/path under this test. If the fast side switches to physical GPU5 (new
CUDA0), process enumeration/allocation order becomes the next focus. A missing
baseline gap, changed CE reference or unstable data needs separate interpretation;
do not force a binary diagnosis. No PCIe configuration change follows from this
test alone.
