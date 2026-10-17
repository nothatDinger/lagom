# PR #40670 H2D reference: restore the baseline before optimizing

Target: https://github.com/sgl-project/sglang/pull/40670 at immutable commit
`70bc9e9a643faa998ce529a38260feee83d3c0c6` (ilovecoding84/sglang).
PR #40760 is a CI change; the user confirmed #40670 is intended.

## Why rebuild the baseline

Local `99cdf3403` adapted the whole-window hash/CLOCK planner, but retained local
fixed hot slots and separate `copy_cache_planned_mla` I/O. Its CUDA file explicitly
says so. It removed PR scratch-copy/rotation behavior. The previous copy CTA,
D2, and tail experiments directly characterize this local split implementation,
not the PR's fused gather/commit path. Preserve all previous results; do not
reinterpret them as PR benchmarks.

The reference now preserves the PR's full swap kernel bodies and C4 transfer
helper, scratch layout and rotation, miss deduplication/admission, and launch
geometry: gather 64 threads with grid `[B, ceil(W*K/64), 1]`; commit 512 threads
with grid `[B,1,1]`. PDL uses the same architecture detection as PR, and optional
miss-plan output is disabled as in the PR coordinator's verify call. No additional
copy kernel is launched. CTA count is not a measurement of simultaneous SM use.

## Scope and source provenance

This is an isolated reference implementation and check entry, not a serving
migration. Existing coordinator, writer scratch/finalize and production defaults
are untouched. Safe service integration must reconcile PR slot rotation and
finalization with local DSpark mappings; replacing just the copy call is unsafe.
The check does not validate complete model inference, acceptance or TPOT.

Three headers are pinned in `csrc/pr40670_reference/`. `manifest.json` records
source paths, normalized source SHA256 and vendored SHA256. Algorithm bodies are
unchanged. The sole header adaptation redirects `hisparse_transfer.cuh`'s include
to its pinned C4 helper instead of the locally modified helper. EOF whitespace
is normalized. Python reference ops retain the original state tuple, transfer
policy, module arguments and wrappers; only JIT cache names/source paths are
redirected. Layout formulas/validation are copied from the same commit, omitting
server-configuration resolution. Current local JIT/FFI utilities and compiler
are reused; this is source/launch fidelity, not a claim of identical binary code
or a reproduction of PR's complete serving environment.

## First GPU job: correctness and configuration only

In a single gpuq allocation of physical GPU4, using the existing venv:

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
git pull --ff-only origin perf/hisparse-h2d-diagnosis
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_hisparse_pr40670_check.sh \
  "$PWD/work/hisparse-pr40670/check-v1"
```

One new absolute output directory. Uses existing CUDA/PyTorch/cuobjdump; no NCU
or hardware-counter permissions needed. CPU and host allocation bind to node1,
Portable registration, physical GPU4 UUID checked. No performance job follows.

Three cases use captured B=4 with three real requests in order [2,0,1] and an
invalid padded request ID. W2/K4/H16 is checked with scratch14 and scratch0;
W6/K16/H128 with scratch13 exercises gather-copy and commit direct-overflow
behavior. Small Top-K is a correctness geometry, not the future service benchmark
configuration. No performance conclusion can be drawn from these cases.

Each has eager validation and six mutable graph replays: cold unique misses,
cross-step duplicates/invalid tokens, warm repeat, mixed hits/misses and per-step
lengths, changed host data after reset, and zero real requests. The oracle checks
every value+scale byte against CPU host content, invalid output locations,
unchanged non-output/guard/padding bytes, and physical slot permutation after
rotation. State and data reset outside capture. It does not assume PR output
slots or cache residency must equal the local split planner's choices.

A separate single graph replay is profiled for each case (PyTorch/Kineto, not
NCU). Exactly two GPU kernel activities with expected names/grid/block/device
are required. An independent copy kernel or shape mismatch fails. Offline
cuobjdump extracts SASS/resources for the two swap specializations. Resource
extraction is for review; the script does not impose old copy-kernel opcode
counts or assume PR gather/commit must have no local memory.

Return `result/reference-compact.txt` in full. Keep `reference.json`, the three
`.trace.json` files and `jit*.sass.txt`/`jit*.resources.txt`. If failed return the
compact plus last 30 lines of run.log and stop. Do not bypass correctness/launch
checks. Passing means this standalone reference is ready for benchmark design,
not that production serving is migrated (`service_integrated=False`) or TPOT is
improved (`performance_ready=False`).

## Subsequent comparison and service integration

After correctness/launch review, compare the entire PR gather+commit interval
against the entire local gather+commit+copy interval using matched input and
initial cache content. Keep each implementation's native state/layout and
record host bytes, unique misses, scratch/admission and residency differences.
Do not time only the old standalone copy or repeatedly warm one snapshot until
it becomes a hit-only workload. Include controlled cold/warm cases, representative
batch/verify/miss distributions, and stable graph replay inputs.

Because admission/physical mapping differs, such a comparison evaluates two
complete implementations; it cannot attribute all time difference to fusion.
Only after contents and state transitions are validated should we migrate the
service's prepare/finalize/memory-sizing together, then measure acceptance,
output correctness, full decode TPOT and service throughput under DSPARK plus
high concurrency. Pause hardware queue localization and further CTA tuning in
the meantime. No new performance or service claim is made in this commit.

## Padded request output contract (harness correction)

The pinned gather kernel writes exactly **0** to all output locations of rows
`bid >= num_real_reqs` before returning, without dereferencing their request IDs.
Inside active request rows, invalid token/sequence occurrences still return -1.
The original oracle incorrectly required -1 in both cases; this was a harness
mismatch. The corrected oracle requires exact zero for padded rows, not either
zero or -1. Padding values never enter the content lookup or permitted-write
mask. A zero-real-request replay requires all output locations zero and all
KV bytes, guards, padding and physical mappings unchanged. All previous byte
and state checks remain enabled. The pinned CUDA bodies/manifest are untouched.

This is an observed baseline behavior, not proof of downstream safety. In the
pinned PR coordinator `swap_in_speculative_pages` returns the full captured
`device_locs.view(-1, top_k)` and the C4 indexer places it in attention metadata;
there is no real-row slicing at that boundary. Whether specific attention
backends process dummy rows, use reserved slot 0, or discard their outputs must
be verified at serving integration. Report `consumer_safety_verified=False`
until then. Do not label a standalone reference pass as a complete service
correctness pass or patch the pinned kernel to make the old oracle pass.
