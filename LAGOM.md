# Lagom: DSPARK decoding with CSA KV prefetch

Lagom extends DeepSeek-V4 DSPARK speculative decoding with a disposable prediction
suffix. The suffix supplies per-layer CSA Top-K hints for the following verify
round while the original candidates retain normal speculative acceptance.

## Window conventions

| Quantity | Meaning |
| --- | --- |
| `W` | Number of draft candidates eligible for acceptance. |
| `P` | Number of additional draft candidates used only for prediction; defaults to `W`. |
| `--speculative-num-draft-tokens` | `W + 1`, including DSPARK's existing anchor token. |
| Draft execution width | `W + P`. |
| Target execution width | `W + P + 1`, including the anchor. |

For example, `--speculative-num-draft-tokens 8` gives `W=7`. With the default
prediction window, the draft generates 14 candidates and the target evaluates 15
rows per request. Only the first seven candidates participate in acceptance.
The scheduler still receives the original eight-column result layout.

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
  --max-running-requests 8 \
  --disable-radix-cache \
  --enable-hisparse \
  --hisparse-config '{"host_to_device_ratio":2,"device_buffer_size":8192,"force_h2d_each_step":false,"prefetcher":"lagom","prefetcher_config":{"prediction_steps":4,"buffer_slots":21}}'
```

Omit `prefetcher_config.prediction_steps` to use `P=W`. It must otherwise be a
positive integer. `prefetcher: "lagom"` selects DSPARK when the speculative
algorithm is omitted; an explicitly different algorithm is rejected. The
existing bundled-DSPARK draft-path default remains available where the target
checkpoint provides it.

The example explicitly sets W=7 and P=4. P remains configurable; omitting it
still selects P=W. Without the decode block-size override,
DSPARK validates the requested `W` against the draft checkpoint's block size.

KV storage has three distinct lifetimes:

- The persistent **hot cache** holds committed history for temporal locality.
  Its budget B is independent of W/P. DSPARK defaults to 2*K entries when B is
  omitted; explicit B must be at least the model's effective Top-K.
- One **verification arena** is shared by serial CSA layers. It materializes
  W+P+1 query selections and stays live through attention and promotion.
  Its union may exceed B; the kernel bound (W+P+1)*K <= 8192 remains.
- A dedicated **circular prefetch buffer** holds N forecasts of at most P*K
  C4 rows per request per slot. It never aliases the hot cache or verification arena.

prefetcher_config.buffer_slots configures N. Omitting it uses the CSA layer
count, which is 21 for this Flash checkpoint. Full-step mode requires N >= that
count: every layer's forecast stays live until the next verify consumes it.
A smaller ring needs rolling scheduling and is rejected in this implementation.
Slot mapping uses the CSA ordinal modulo N, not the transformer's absolute layer ID.

At W=7/P=4/K=512, the shared verification capacity is 6144 rows. N=21 adds about
24 MiB of forecast payload per request per GPU, or 6 GiB at 256 requests, before
padding and metadata. The startup memory budget includes the shared arena,
all N payload slots, indices and per-layer forecast descriptors.

B counts compressed C4 KV rows, not original tokens. The model's index_topk
determines K. The example keeps B=8192 for comparison with existing runs; B
does not need to contain the verification union. If configuring B below the
legacy top_k default, also set top_k to the model's actual index_topk so the
generic configuration validator uses the intended minimum.

Optional `prefetcher_config.size` limits the number of selected C4 entries from
each prediction row. It defaults to `K` and must be between 1 and `K`. Reducing
it changes the hints, not the Top-K used by real attention. Duplicate entries
across prediction rows are resolved by the fused cache planner.

Keep `force_h2d_each_step` false: invalidating residency on every step would
discard historical residency and is rejected in Lagom mode.

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

1. DSPARK proposes W+P candidates. The target executes the anchor plus all
   candidates; acceptance still considers only the first W.
2. Each CSA layer waits for its previous forecast transfer, then the indexer
   produces current and future Top-K selections.
3. The resolver deduplicates the query union into one shared verification arena.
   Hot/forecast hits and tentative C4 writer rows use D2D copies; unresolved
   real demand uses H2D. Forecast-only host misses are omitted from this round's
   approximate prediction attention.
4. Attention reads the materialized arena. Newly generated speculative C4s
   remain in independent writer scratch until acceptance/rollback.
5. After attention, selected committed historical rows may be copied back to
   stable hot-cache addresses. Prediction-only rows and tentative new C4s
   cannot populate the hot cache. The verification arena is now reusable.
6. Attention completion and promotion release this layer's forecast slot.
   Only then does the side stream replace its consumed bytes with next-round
   predicted KV. Existing hot hits need no forecast H2D.
7. These copies overlap later layers and the following draft run. The next
   verify waits at each consuming layer. Request release drains writers before
   host pages or request slots can be reclaimed.

The full-step mode retains forecasts across the verify boundary in N independent
payload slots. Acceptance and C4 commit/rollback retain the original W boundary.
The design follows the separation of hot residency and reusable verification
staging in [DHiSparse, sections 3.3/3.4](https://arxiv.org/pdf/2610.13708v1).

To preserve FlashMLA's single C4-cache-base interface, this implementation
materializes hot and forecast hits into the shared arena with D2D copies.
Their cost and the post-attention copy-back should be measured separately from
demand H2D; byte correctness and a lower HBM requirement do not establish a
performance improvement.

## CUDA graphs

CUDA graphs remain enabled by default for both target and draft. Their buffer
and capture geometry use the physical execution widths: `W+P+1` for target and
`W+P` for draft. Acceptance and scheduler transport retain `W+1`.

The target graph captures planning, materialization, attention, historical
promotion and each layer's external slot-release event. The side transfer graph
is enqueued immediately after target launch and waits for these release events.
There is no target-entry global prefetch wait or foreground graph-tail join.

The verification arena/plan and per-layer forecast descriptors/live counts have
fixed addresses. Each layer waits for its prior forecast copy before reading or
overwriting its descriptors. Padded requests cannot mutate payloads or indices.
Changing W/P/N requires restarting so allocations and captures are rebuilt.

Use `--disable-cuda-graph` with the same launch command for an eager comparison.


## Mooncake PD disaggregation

Lagom may run on a CUDA DeepSeek-V4 **decode** server with Mooncake. The
prefill server remains ordinary DSPARK: enable DSPARK on both nodes, enable
HiSparse/Lagom only on decode, and pass the decoder's execution width to prefill:

```text
prefill:
  --disaggregation-mode prefill
  --speculative-algorithm DSPARK
  --speculative-num-draft-tokens W+1
  --speculative-dspark-block-size W
  --disaggregation-dsv4-state-execution-width W+P+1

decode:
  --disaggregation-mode decode
  --speculative-algorithm DSPARK
  --speculative-num-draft-tokens W+1
  --speculative-dspark-block-size W
  --enable-hisparse
  --hisparse-config ... "prefetcher":"lagom",
                      "prefetcher_config":{"prediction_steps":P,"buffer_slots":21} ...
```

The expressions above must be replaced by integers in manual launch commands.
The `perf/lagom` overall launcher computes them automatically from its `w`/`p`
configuration. With W=7/P=4, the public verify width stays 8, the layout width is
12 and both nodes use a 32-slot C4 state ring. With P=7, layout width is 15 and
the ring is also 32; smaller windows can use 16. No runtime rule fixes P to 4.

The prefill-only option changes attention/indexer compressor state sizing,
memory budgeting and the matching kernel addressing. It does not enable
HiSparse/prefetch on prefill, change its proposal width, or extend its public
acceptance window. Decode keeps its existing W/P execution and CUDA Graph
path. The runtime default remains P=W when prediction_steps is omitted.

Both nodes must use matching target/draft checkpoints, equal TP and matching
state formats; PP/DP/CP/DCP remain 1 for this path. Use Mooncake on both sides.
NIXL/fake transfer and enabling Lagom prefetch on the prefill node remain
unsupported. Changing W/P requires restarting the paired servers so their
state allocations and CUDA graphs are rebuilt consistently.

Mooncake now checks the selected state buffers' item sizes against the receiver
metadata before issuing state copies. A missing prefill option or a different
ring size fails the request with a layout mismatch instead of writing with the
wrong destination stride. This compares per-item layouts, not whole-pool
capacities; the two sides can reserve different numbers of pages. Existing
metadata is sufficient; there is no new wire field or per-verify transfer
repacking.

CPU regression coverage includes W/P-dependent ring sizing, exported attention
and indexer state strides, nonzero destination pages, ring wrap, and refusal of
the old 16-to-32 layout before DMA:

```bash
python -m unittest discover -s test/registered/unit/disaggregation \
  -p test_lagom_pd_state_layout.py
python -m pytest -q test/registered/unit/spec/test_lagom_config.py
```

This compatibility change was prepared without a usable execution environment;
the CPU tests and GPU smoke tests have not been run here. Before collecting
performance results, exercise P=1/2/4/7, prefix lengths around ring/page
boundaries, mixed batch sizes, request-slot reuse and retract/rebootstrap with
CUDA Graph enabled. Verify the first decode step after transfer as well as
subsequent verify/prefetch steps.

## Supported configuration

The current path targets CUDA DeepSeek-V4, DSPARK, HiSparse, static verify
windows, and either colocated prefill/decode or Mooncake PD decode with a
layout-compatible DSPARK prefill server. Lagom selects static verify internally;
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
  test/registered/unit/mem_cache/test_hisparse_spec_swapin.py \
  test/registered/unit/mem_cache/test_hisparse_staging_layout.py
```

CPU coverage exercises configuration, request-major window boundaries, cache
ownership, C4 commit boundaries, split-transfer plans, graph launch, and stream
dependencies with tensors/mocks. The graph test also includes a CUDA runtime
case that is skipped without a GPU.
These checks do not establish full-model GPU correctness, numerical parity, or
throughput.

On a CUDA machine, also exercise union planning, demand/prefetch ownership,
concurrent attention reads, resident lookup, and repeated graph/side-stream
cache updates against reference KV bytes:

```bash
python -m pytest -q test/registered/kernels/ops/kvcache/test_hisparse_staging.py
python -m pytest -q test/registered/kernels/ops/kvcache/test_hisparse_spec.py
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
that waits occur at the consuming layer. Measure latency, demand H2D bytes,
cache hits, and acceptance separately. This implementation does not include a
GPU performance or prediction-accuracy claim.


The staging regressions cover P=1/2/4/7, unions larger than B, packed values and
quantization scales, all 21 independent forecast slots across successive graph
replays, and request-slot reuse. Run the plain-Python allocation-accounting test:

~~~bash
python -m unittest discover -s test/registered/unit/mem_cache -p test_hisparse_staging_layout.py
~~~

This change was authored without a usable execution environment. These tests
have not been run here; validate them and the model/PD smoke cases before using
the new implementation for performance conclusions.

For a short, reproducible three-mode throughput comparison and CUDA Graph
per-step timing, see [the performance experiment guide](benchmark/lagom/README.md).
