# Lagom FP: CSA prefetch from frozen first-layer inputs

The `lagom_FP` branch extends DeepSeek-V4 DSPARK decoding with a separate
prediction stream. The draft proposes W acceptance candidates and P prediction
tokens. Only the anchor and W candidates enter target attention. P tokens retain
their first-layer input hidden states (target token embeddings, before mHC or
decoder transformations); every CSA layer uses that same frozen tensor with its
own query projection and indexer to predict KV needed by the next window.

## Window conventions

| Quantity | Meaning |
| --- | --- |
| `W` | Number of draft candidates eligible for acceptance. |
| `P` | Number of additional draft candidates used only for prediction; defaults to `W`. |
| `--speculative-num-draft-tokens` | `W + 1`, including DSPARK's existing anchor token. |
| Draft execution width | `W + P`. |
| Target attention/FFN width | `W + 1`, including the anchor. |
| Prediction indexer width | `P`, on a separate CUDA stream. |

For example, `--speculative-num-draft-tokens 8` gives `W=7`. With the default
prediction window, the draft generates 14 candidates. The target decoder evaluates
8 rows per request, while each CSA indexer separately evaluates 7 prediction
queries from the frozen first-layer input. Only the first seven draft candidates
participate in acceptance. The scheduler retains its eight-column result layout.

The split is applied independently to each request. Target logits, proposal
probabilities, confidence, hidden-state injection, and committed KV lengths all
use the original acceptance boundary. Prediction tokens never become output
tokens or committed draft context merely because they were evaluated.

## Enable Lagom

Use a DeepSeek-V4 target and a compatible DSPARK draft checkpoint. Set the
checkpoint paths and tensor-parallel size for the available GPUs:

```bash
python -m sglang.launch_server \
  --model-path /path/to/deepseek-v4 \
  --speculative-draft-model-path /path/to/compatible-dspark \
  --speculative-algorithm DSPARK \
  --speculative-num-draft-tokens 8 \
  --speculative-dspark-block-size 7 \
  --tp-size 8 \
  --disable-radix-cache \
  --enable-hisparse \
  --hisparse-config '{"host_to_device_ratio":2,"device_buffer_size":8192,"force_h2d_each_step":false,"prefetcher":"lagom","prefetcher_config":{"prediction_steps":7}}'
```

Omit `prefetcher_config.prediction_steps` to use `P=W`. It must otherwise be a
positive integer. `prefetcher: "lagom"` selects DSPARK when the speculative
algorithm is omitted; an explicitly different algorithm is rejected. The
existing bundled-DSPARK draft-path default remains available where the target
checkpoint provides it.

The example explicitly sets the decode block size to 7. Without that override,
DSPARK validates the requested `W` against the draft checkpoint's block size.

`device_buffer_size` counts compressed C4 KV entries, not original tokens. The
model's `index_topk` supplies `K`; the legacy HiSparse `top_k` field does not
override the DeepSeek-V4 indexer. If the device buffer size is omitted, it
defaults to `2 * max(W+1, P) * K`. An explicit size must hold at least
`(W+P+1) * K` entries. The fused planner additionally requires this working
set to be at most 8192 entries and the hot buffer to be at most 32768 entries.
The example above assumes the model's usual `K=512`.

Demand loading and prefetch indices are still planned together at the full
`W+P+1` width, even though target attention only has `W+1` rows.
The cache must protect both sets simultaneously because their transfers run
concurrently. The default buffer size is unchanged, but explicit sizes that
only held the larger of the two windows are no longer sufficient. Pool sizing
also reserves the persistent per-layer split-transfer plans and live-request
counts used by CUDA Graph replay.

Optional `prefetcher_config.size` limits the number of selected C4 entries from
each prediction row. It defaults to `K` and must be between 1 and `K`. Reducing
it changes the hints, not the Top-K used by real attention. Duplicate entries
across prediction rows are resolved by the fused cache planner.

Keep `force_h2d_each_step` false: invalidating residency on every step would
discard the prefetched working set and is rejected in Lagom mode.

## Run DSPARK + HiSparse without Lagom

Omit `prefetcher` and `prefetcher_config` to disable Lagom. Keep
`--speculative-algorithm DSPARK --enable-hisparse --disable-radix-cache` and the
same model, window and cache-capacity arguments, replacing the HiSparse config
with:

```bash
--hisparse-config '{"host_to_device_ratio":2,"device_buffer_size":8192}'
```

`force_h2d_each_step` defaults to `false` in both modes. Ordinary DSPARK uses W
draft candidates, preserves resident KV across steps, and has no prediction
suffix or Lagom prefetch transfers. Set `SGLANG_RAGGED_VERIFY_MODE=static` for
the ordinary DSPARK baseline to retain target CUDA Graph; Lagom selects static
verify automatically. This gives a comparison with the same residency policy.

Explicit `force_h2d_each_step=true` remains available for forced-cold baseline
experiments. It requires a compressed prefix longer than the hot buffer and
cannot be combined with Lagom.

## Execution and stream ordering

1. DSPARK proposes `W+P` candidates. The target receives only the anchor and W
   candidates as attention rows. P token IDs travel through a separate
   request-major input buffer.
2. The target embeds P once and preserves that first-layer input. At target
   entry, a prediction stream starts all CSA layers' query projections and
   indexer Top-K using this unchanged tensor. Prediction positions follow the
   W window, but searchable indexer keys are restricted to committed history.
   Prediction never invokes a compressor, attention, FFN, or target KV write.
3. The real indexer waits for its layer's prediction result, then a single
   planner protects the union of demand and prediction selections. Shared
   misses belong to demand; prediction-only misses get distinct destinations.
   Only demand locations are returned to attention.
4. The main path refills demand misses while the transfer stream prefetches
   prediction-only misses for the next window. This starts during the current
   verify, before acceptance or C4 commit. Predictions affect residency only;
   the next window's real indexer still determines its own attention selection.
5. Acceptance and C4 transactions cover only the anchor and W candidates.
   Rejected real rows are rolled back, and unaccepted C128 state is cleared.
   P has no target KV or hidden-state outputs to commit. Allocation bounds
   remain conservative at `W+P+1` to accommodate the draft and union planner.
6. Each layer waits for its preceding prefetch before cache-plan reuse. Request
   release drains both prediction readers and outstanding transfers before
   storage reuse. Error rollback also fences prediction work that may not yet
   have reached the normal per-layer join.

Demand loading and prefetch share the persistent hash/CLOCK state from
`swapin_oneshot`. Prediction accuracy affects cache hit rate, not the real
attention selection: the next real indexer still chooses its own Top-K, and
missing entries use ordinary demand loading. Hints remain useful when the
actual accepted length differs from `W`; they do not assume that the complete
draft path was accepted.

## CUDA graphs

CUDA graphs remain enabled by default for both target and draft. Target attention,
logits, and auxiliary hidden states capture `W+1` rows; draft capture keeps
`W+P`. A separate graph-stable `[batch, P]` input buffer supplies predictions,
including on pre-planned replay. Its padded request rows are cleared each time.

The prediction indexer work is captured on its own CUDA stream and joins the
real indexer at the union planner. It uses live device positions and page tables
on every replay. It shares no writable attention metadata with the real indexer.

The target graph retains stable per-layer transfer plans and records a plan-ready
event at each CSA layer before demand refill. A separate side-stream transfer
graph is launched immediately after the target graph is enqueued. Each transfer
waits on its layer's readiness event, allowing H2D to run while the target graph
is still executing that verify. Live-request counts exclude padded graph rows;
committed source lengths are derived from graph-updated device inputs.

Persistent external CUDA events provide per-layer wait nodes inside the target
graph. Side-stream prefetch records those events for the following replay;
there is no global prefetch wait at target entry or foreground graph-tail join.
The fused acceptance/commit epilogue is not used for Lagom because its original
single-width buffers would mix the decoding and prediction boundaries.

Use `--disable-cuda-graph` with the same launch command for an eager comparison.

## Supported configuration

The current path targets CUDA DeepSeek-V4, DSPARK, HiSparse, static verify
windows, and colocated prefill/decode. Lagom selects static verify internally;
no `SGLANG_RAGGED_VERIFY_MODE` override is needed. Tensor parallelism is supported;
PP, DP, attention CP, and DCP sizes must remain 1. Adaptive speculation and the offline
`SGLANG_OASISKV_TRACE_CONFIG` harness cannot be combined with Lagom. Grammar or
constrained decoding remains unsupported by the DeepSeek-V4 HiSparse DSPARK
verify path. Online C128 speculative compression retains its existing
EAGLE-only restriction.

The full execution width must also fit the compressed-state ring and the
request-owned transfer scratch. Startup validates these bounds. The C4 ring
grows from 16 to the required power of two, up to 128, so the default `W=7,P=7`
uses a 32-slot ring rather than overwriting state needed by the next verify.

## Validation

With the repository's Python test dependencies installed, run from the
repository root:

```bash
python -m pytest -q \
  test/registered/unit/spec/test_lagom_config.py \
  test/registered/unit/spec/test_lagom_windows.py \
  test/registered/unit/models/test_lagom_csa.py \
  test/registered/unit/managers/test_lagom_prefetch.py \
  test/registered/unit/managers/test_lagom_commit_lifecycle.py \
  test/registered/unit/spec/test_lagom_cuda_graph.py \
  test/registered/unit/spec/test_lagom_graph_startup.py \
  test/registered/unit/managers/test_hisparse_dspark.py \
  test/registered/unit/mem_cache/test_hisparse_spec_swapin.py
```

CPU coverage exercises configuration, request-major W/P separation, frozen
prediction inputs, query-only indexer behavior, cache ownership, C4 commit
boundaries, split-transfer plans, graph inputs, and stream dependencies with
tensors/mocks. The graph test also includes a CUDA runtime
case that is skipped without a GPU.
These checks do not establish full-model GPU correctness, numerical parity, or
throughput.

On a CUDA machine, also exercise union planning, demand/prefetch ownership,
concurrent attention reads, resident lookup, and repeated graph/side-stream
cache updates against reference KV bytes:

```bash
python -m pytest -q test/registered/kernels/ops/kvcache/test_hisparse_spec.py -k lagom
```

For GPU validation, run the same prompts with graphs enabled and with
`--disable-cuda-graph`, covering:

- `P=1`, `P=W`, and `P>W` within capacity, with multiple requests and changing
  batch sizes, including padded graph buckets.
- Rejection at the first candidate, partial acceptance, and complete
  acceptance of all W candidates; committed output and lengths must never
  include prediction-only rows.
- Prefix lengths around C4, C128, and SWA/page boundaries, plus short prefixes
  and contexts larger than the resident device buffer.
- Request completion, cancellation, slot reuse, and errors with prefetch
  outstanding; inspect for stale IDs, invalid locations, and transfer races.
- Greedy output against a complete-KV target reference. For stochastic
  sampling, test acceptance/distribution behavior rather than assuming
  identical RNG consumption when additional draft candidates are generated.

Use a CUDA profiler to check that prediction prefetch runs on the side stream
concurrently with the source verify, including with CUDA Graph enabled, and
that waits occur at the consuming layer. Check that P has only indexer work on
the prediction stream and that all target attention/FFN kernels use W+1 rows. Measure latency, demand H2D bytes,
cache hits, and acceptance separately. This implementation does not include a
GPU performance or prediction-accuracy claim.
