# Standalone fused HiSparse swap-in

`swapin_oneshot` starts directly at `heykon_main@aaff55ab6`, the parent of
`perf/hisparse-h2d-diagnosis`'s first exclusive commit (`881a02074`).

DeepSeek-V4 speculative swap-in uses `pr_fused` directly. There is no
`SGLANG_HISPARSE_VERIFY_IMPL` selector or split implementation in this branch.
The gather/commit pair updates the complete verify window's hash/CLOCK state,
deduplicates misses, and transfers KV. Single-token fallback uses the same state.

The extracted kernel is the PR #40670 implementation pinned at `70bc9e9a`, with
the CAS retry correction from `cbaea63b2`. It was connected to serving in
`3f9ff3a49`. Kernel bodies and launch shapes are preserved; unused PR accepted-C4
finalization kernels are omitted because the parent branch already owns those
transactions in `HiSparseCoordinator`. The source branch's benchmark imports,
manifest checks, split planner, probes and comparison scripts are not needed.

The original serving restrictions are retained: CUDA DSv4 speculative verify,
`force_h2d_each_step=true`, real KV IO, no named prefetcher, compressed prefixes
longer than the hot buffer, and `verify_width * effective_top_k <= min(hot, 8192)`.
For DSPARK, the hot buffer defaults to `2 * verify_width * effective_top_k`,
including force-H2D mode (width 6 and effective Top-K 512 default to 6144 C4
entries). Explicit buffer sizes are preserved if they can hold one complete
verify window; force-H2D invalidates residency without changing that capacity.
The canonical tail and provisional C4 placeholders remain fixed. Remaining slots
in each request's existing allocation provide rotating KV scratch; separate
coordinator-owned C4 writer scratch, accepted-token commit and rollback retain
the parent branch's implementation. Hash/CLOCK metadata is budgeted before KV
pool allocation. No backend environment variable is required.

Run CPU regression checks from the repository root:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_hisparse_spec_swapin.py \
  test/registered/unit/managers/test_hisparse_dspark.py \
  test/registered/unit/managers/test_hisparse_eagle.py \
  test/registered/unit/test_hisparse_prefetcher.py
```

On a CUDA machine, also run the KV-content and graph-replay tests:

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hisparse_spec.py
```

The CUDA tests exercise non-monotonic request IDs, padding, duplicate and invalid
selections, scratch overflow, repeated forced-cold rotations, all 584 KV bytes,
preservation of the request's allocation set and canonical/provisional slots,
and CUDA graph replay. A skipped CUDA suite is not a CUDA correctness pass.
