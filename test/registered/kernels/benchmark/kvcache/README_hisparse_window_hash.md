# HiSparse whole-window hash latency

Run these measurements on NVIDIA CUDA with this branch installed. DeepSeek V4
uses 584 bytes per C4 entry (576 value bytes and 8 scale bytes). `--top-k` in the
microbenchmark is the effective C4 Top-K, normally 512 for V4-Flash; it is not
the compatible original-token `top_k` value in `--hisparse-config`.

## Measure plan and IO with CUDA events

From the repository root:

```bash
python test/registered/kernels/benchmark/kvcache/bench_hisparse_window_hash.py \
  --batch-size 4 --verify-width 6 --top-k 512 \
  --miss-rate 0.2 --overlap 0.75 --mode all \
  --output-file plan-io.json
```

Both planners receive the same initial resident entries, Top-K window, physical
slot mapping and DSv4 copy kernel. The JSON contains `lru` and `window_hash`
results with mean, median, p95 and minimum microseconds per layer per complete
batch verify window, unique miss counts and actual H2D bytes. Miss rate describes
each row relative to the initial cache; overlap describes repeated misses across
rows. Copying uses 4 CTAs by default, matching the runtime's synchronous verify
path; adjust `--num-copy-blocks` when comparing another runtime configuration.

- `--mode plan-io` (default): planning followed by copying the unique miss union.
- `--mode metadata-only` or `--metadata-only`: planning without the copy kernel.
- `--mode copy-only`: replay an already computed plan, excluding planning.
- `--mode all`: measure all three independently. Their times are not assumed to
  add exactly because launch overhead and cache conditions differ.
- `--timing cuda-graph` (default): CUDA events surround one graph replay.
- `--timing eager`: CUDA events surround direct launches, like main's
  `bench_hisparse.py`. Host submission gaps can contribute to this measurement.
- `--skip-io`: dispatch the copy kernel with its KV-byte transfer disabled. This
  also follows `SGLANG_DEBUG_HISPARSE_SKIP_IO=1`, like the server. Use this with
  `--mode plan-io` to preserve copy dispatch overhead in the IO ablation.

The initial snapshot reset is outside timing, unlike main's original fused MLA
microbenchmark, which also times its two snapshot copies. Force-H2D invalidation
is part of the timed plan because the runtime performs it on each step:

```bash
python test/registered/kernels/benchmark/kvcache/bench_hisparse_window_hash.py \
  --batch-size 4 --verify-width 6 --top-k 512 --mode plan-io \
  --force-h2d-each-step --timing eager --output-file force-plan-io.json
```

Force-H2D sets the effective hot capacity to `verify_width * top_k`, as the server
does, even if `--hot-slots` is supplied. It forgets the preceding step's resident
identities and rebuilds the hash once before the complete window, preserving
physical slots. All distinct valid selected entries must be copied again;
repeated tokens within the current window still share one copy. The benchmark
checks both planners' miss counts against this expected union.

## Enable force-H2D in the server

Keep your existing V4/DSPARK model, draft model, TP/DP and launch configuration.
Include the following in its HiSparse arguments:

```bash
--enable-hisparse \
--hisparse-config '{"top_k":2048,"host_to_device_ratio":5,"swap_in_block_size":960,"force_h2d_each_step":true}'
```

The model's effective `index_topk` controls K. With DSPARK the hot capacity is
`verify_width * effective_top_k`, including in force-H2D mode. The flag is a JSON
boolean. Omit it or set it to `false` to retain entries between steps.

For the skip-IO comparison, restart the same server with
`SGLANG_DEBUG_HISPARSE_SKIP_IO=1`. For the normal run, set it to `0`. Keep
`force_h2d_each_step` and all other server settings identical in the two runs.
Skip-IO retains planning and metadata updates, but does not produce valid model
output. In speculative decoding it can also change acceptance, so record the
acceptance lengths and interpret the difference as an end-to-end ablation.

## Measure TPOT and collect a decode trace

Use your actual model path or Hugging Face identifier for `MODEL_PATH`, and the
running server URL for `BASE_URL`. Keep the same hardware, TP/DP configuration,
prompt seed, lengths and concurrency across runs. Use a long enough prompt to
exercise host-backed C4 entries instead of the short-sequence path.

```bash
python -m sglang.benchmark.serving \
  --backend sglang --base-url "$BASE_URL" --model "$MODEL_PATH" \
  --dataset-name random --num-prompts 16 --max-concurrency 4 \
  --random-input-len 32768 --random-output-len 512 \
  --random-range-ratio 0 --seed 42 --warmup-requests 4 \
  --output-file normal.jsonl
```

Repeat the same command against the restarted skip-IO server, changing only
the result filename to `skip-io.jsonl`. The default benchmark ignores EOS to
keep output lengths fixed. Files are append-only; the reporter uses their last
row unless you set `--result-index`.

Collect a separate profile run on the **normal IO** server so profiler overhead
does not contaminate the normal TPOT reference. Add these serving benchmark
flags to record a bounded sample:

```bash
--profile --profile-start-step 20 --profile-steps 30 \
--profile-output-dir /tmp/hisparse-profiles --profile-prefix hisparse
```

Inspect the exported trace and choose a steady decode-only interval with fixed
batch size and the same verify geometry as the TPOT run. `profile-start-step`
is a scheduler forward counter, not a guaranteed number of decode steps to
skip; startup/warmup can already have advanced it. Use `--trace-start-us` and
`--trace-end-us` below if the trace contains prefill or changing batches. Count
the actual scheduler decode/verify steps in the chosen interval. If CUDA graphs
are enabled, the trace must contain individual GPU graph-node kernels; a
`cudaGraphLaunch` entry alone is insufficient.

## Report the TPOT shares

Set `TRACE_PATH` to one rank's `.trace.json` or `.trace.json.gz`. Do not concatenate
the times from all tensor-parallel ranks. `DECODE_STEPS` is the number of complete
profiled decode iterations; `TOKENS_PER_STEP` is the average committed output
tokens **per request per iteration** in that same interval. Use 1 for ordinary
decode. For DSPARK use measured committed output, including the bonus token;
neither verify width nor proposed draft count is an output-token count. Do not
divide by batch size again.

```bash
python -m sglang.benchmark.hisparse_latency \
  --normal-result normal.jsonl --skip-io-result skip-io.jsonl \
  --trace "$TRACE_PATH" --device 0 \
  --decode-steps "$DECODE_STEPS" --tokens-per-step "$TOKENS_PER_STEP" \
  --output-file hisparse-tpot.json
```

The report keeps two meanings separate:

1. `plan_io_gpu_share_of_tpot_pct` uses actual gather, commit and copy kernel
   durations from the normal trace. It merges overlapping HiSparse intervals
   on the selected GPU, divides by decode steps and committed tokens per request,
   and then divides by normal TPOT. The report also supports main's fused
   swap-in kernel. This is GPU time occupied by plan+IO, not the TPOT saved by
   removing it. It excludes launch gaps and residency invalidation; hash
   initialization appears as a separate kernel statistic. The force-H2D
   microbenchmark above includes the full invalidation cost.
2. `exposed_io_share_pct = (normal_TPOT - skip_IO_TPOT) / normal_TPOT * 100`.
   Planning remains in both runs. This measures the end-to-end effect of IO,
   including overlap and any acceptance change, rather than the entire plan+IO
   cost. Negative differences are retained instead of being clamped.

You can omit `--skip-io-result` for a trace-only report, or omit the trace and its
normalization arguments for an ablation-only report. The reporter rejects
mismatched saved workloads and server HiSparse configurations. Repeat matched
runs to assess variance before drawing conclusions from a small difference.
