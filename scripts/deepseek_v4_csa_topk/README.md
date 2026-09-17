# DeepSeek V4 HiSparse + DSpark CSA Top-K experiment

## Design

The controlled variable is CSA `top_k`: 512, 1024, 2048, and 4096. Every group
uses the same single 110,000-token random request by default, seed 0, and
concurrency 1. The long request is intentional: it exceeds the 98,304-token
threshold at K=4096, so all four groups exercise H2D rather than returning an
all-zero curve. The DSpark verify width is
`DSPARK_BLOCK_SIZE + 1` (the 0731 checkpoint default is 5 + 1); consequently the resident HiSparse buffer is set to
`verify_width * K`, the minimum safe size for a verify window whose Top-K sets
are disjoint.

`DETERMINISTIC_INFERENCE` defaults to 0 and must remain disabled for this model
on the current SGLang version. DeepSeek V4 unconditionally selects the `dsv4`
attention backend, while deterministic inference accepts only `ascend`, `fa3`,
`fa4`, `flashinfer`, and `triton`. The runner rejects a true value before model
startup with an actionable error instead of consuming a GPU allocation and then
failing in server-argument resolution. Do not add
`--enable-deterministic-inference` through `SERVER_EXTRA_ARGS`.
The server always receives `--disable-radix-cache`, which is a mandatory
HiSparse constraint; omitting it causes SGLang argument validation to fail
before the model workers start.

DeepSeek-V4-Flash-0731 contains FP4 MoE weights. The experiment therefore uses
`MOE_RUNNER_BACKEND=flashinfer_mxfp4` by default, matching the official 0731
launch recipe. Letting this checkpoint fall back to the Triton MoE runner can
fail during CUDA graph capture with `AssertionError: Hidden size mismatch`.
Override the variable only when the selected checkpoint and installed backend
are known to use a different compatible weight layout.

The runner sets `MEM_FRACTION_STATIC=0.85` rather than allowing the generic
GPU-memory heuristic to fill nearly all of an 80 GiB device with weights and KV
cache. This leaves several GiB for the temporary workspace allocated by
FlashInfer's fused-MoE kernel. If `cutlass_fused_moe` still reports an
out-of-memory error, lower `MEM_FRACTION_STATIC` (for example, to `0.82`). This
reduces KV capacity, so do not lower it farther than necessary for the longest
prompts in the corpus.

Because the benchmark fixes `--max-concurrency 1`, the server also receives
`--cuda-graph-max-bs-decode 1` by default. Capturing the generic 80 GiB-device
default up to batch size 512 creates graph-private pools that this experiment
can never use and may exhaust memory before K=512 starts. Override
`CUDA_GRAPH_MAX_BS_DECODE` only if the benchmark concurrency is raised. The
runner deliberately does not force `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
the observed graph-capture failure had only about 115 MiB reserved-but-unused
but 4.32 GiB in graph-private pools, so it was capacity exhaustion rather than
allocator fragmentation.

### Deterministic mode and the MXFP4 runner

The reported error is an attention-backend incompatibility, not an MXFP4 MoE
runner failure. `flashinfer_mxfp4` remains the correct runner for the 0731 FP4
weights and dispatches separately on SM90, SM100, and SM120. On SM90 it requires
FlashInfer's mixed-input MXFP4 helpers (PR #3084, version 0.6.11 or newer).
Deterministic inference cannot be enabled merely by changing the MoE runner;
SGLang must first add deterministic support for the specialized `dsv4`
attention backend.

There are two server launches per K:

1. **perf** uses normal CUDA graphs and supplies the reported mean TPOT from
   `sglang.benchmark.serving`.
2. **trace** disables CUDA graphs and enables the intrusive H2D probe. The probe
   times each layer's cache-miss copy with CUDA events and records actual miss
   entries per request. Its synchronized TPOT is deliberately discarded. The
   client result is still written to `benchmark_trace.jsonl`; explicitly giving
   both passes an absolute output path prevents `bench_serving` from attempting
   to create its default `sglang_<date>_*.jsonl` in gpuq's read-only working
   directory.

Mean H2D latency is the mean, over decode steps, of the sum of all layer copy
times. `H2D / TPOT` uses milliseconds divided by milliseconds. Each verify
transaction now emits a commit record containing its accepted-token count. The
transfer curve uses the cumulative accepted tokens within that request as its
x-axis, rather than incorrectly treating the decode-step number as acceptance.
It reports physical copies summed across C4 layers in token-layer entries; the
CSV also reports the per-layer mean in logical C4 tokens.

Commit records are also written when a verify window needs no scratch mapping
(for example, when every selected entry is already resident). This keeps every
H2D cycle paired with acceptance data; traces produced before this fix can lack
those records and must be regenerated.

If every `h2d_tokens_per_request` value is zero, first check prompt geometry.
The HiSparse kernel deliberately takes a zero-copy fast path whenever the
compressed C4 sequence length is no larger than `device_buffer_size`. With
DSpark this experiment sizes the buffer to `(DSPARK_BLOCK_SIZE + 1) * K`, and
one C4 entry represents four original tokens. With the default block size 5,
cache misses therefore require sequences longer than roughly `24 * K` original
tokens: 12,288 for K=512, 24,576 for K=1024, 49,152 for K=2048, and 98,304 for
K=4096. The default `DATASET_NAME=random`, `RANDOM_INPUT_LEN=110000` workload
satisfies all four thresholds. Setting `DATASET_NAME=sharegpt` restores the
legacy sampled ShareGPT workload, but ordinary conversations legitimately
produce all zeros unless the supplied corpus contains prompts above these
thresholds.

The analyzer now sums miss counts from every layer rather than reading layer 0
only. It adds a `Sampling diagnostics` warning to `REPORT.md` when all measured
requests fit in the resident buffer. New traces additionally record
`device_buffer_size` and compressed sequence lengths for direct verification.

## Run the analyzer

Run the analyzer from the repository root after the trace run has finished.
Pass the timestamped run directory (or the `latest` symlink), not its parent:

```bash
python3 scripts/deepseek_v4_csa_topk/analyze.py \
  --results-dir results/deepseek_v4_csa_topk/latest
```

The input directory must contain one subdirectory per Top-K value. For example,
a complete default run has this layout:

```text
results/deepseek_v4_csa_topk/latest/
├── k512/benchmark.jsonl
├── k512/h2d_trace.tp0.jsonl
├── k1024/benchmark.jsonl
├── k1024/h2d_trace.tp0.jsonl
├── k2048/benchmark.jsonl
├── k2048/h2d_trace.tp0.jsonl
├── k4096/benchmark.jsonl
└── k4096/h2d_trace.tp0.jsonl
```

To analyze only selected completed groups, list them after `--ks`:

```bash
python3 scripts/deepseek_v4_csa_topk/analyze.py \
  --results-dir results/deepseek_v4_csa_topk/latest \
  --ks 512 1024 2048
```

The command writes or replaces these files inside the selected run directory:

- `REPORT.md`: summary table, incomplete groups, and sampling diagnostics;
- `summary.csv`: mean H2D latency, TPOT, and their ratio;
- `h2d_tokens_by_accepted_tokens.csv`: transfer data by cumulative accepted
  tokens;
- `h2d_tokens_by_accepted_tokens.svg`: plot referenced by `REPORT.md`.

Inspect all command-line options with:

```bash
python3 scripts/deepseek_v4_csa_topk/analyze.py --help
```

If the command reports `trace lacks commit acceptance records`, the trace was
created with old instrumentation. The analyzer cannot reconstruct acceptance
counts from that file: update to the current code and rerun the **trace pass**
(running `gpuq_entry.sh`/`run.sh` creates a fresh run), then analyze the new run
directory. Rerunning only `analyze.py` against the old trace will produce the
same error.

## Run through gpuq

From the repository root, copy or source `env.example`, set `MODEL_PATH`, and
submit the single entry point with your site's gpuq syntax:

```bash
source scripts/deepseek_v4_csa_topk/env.example
# Example only; gpuq flags differ by cluster:
gpuq scripts/deepseek_v4_csa_topk/gpuq_entry.sh
```

`MODEL_PATH` must point to DeepSeek-V4-Flash-0731, whose bundled DSpark draft
head is loaded from the same checkpoint; do not set
`--speculative-draft-model-path`. `DATASET_PATH` is required only when
`DATASET_NAME=sharegpt`.
`SERVER_EXTRA_ARGS` is the supported way to add hardware/checkpoint
specific SGLang flags without editing the experiment. Run one gpuq allocation
with enough GPUs for `TP_SIZE`; do not run the four groups as independent jobs,
because sequential execution keeps the machine and software environment fixed.

Every invocation creates a new directory named
`YYYYmmddTHHMMSSZ_det_off` under
`$RESULTS_DIR/deepseek_v4_csa_topk`. `RESULTS_DIR` defaults to the repository's
`results` directory and may be set to an absolute output directory before the
job is submitted. The `latest` symlink points to the newest run.
If two jobs use the same timestamp and mode, a numeric suffix prevents overwrite.
`RUN_TIMESTAMP` can be supplied by a job scheduler to override the UTC timestamp,
and `RESULTS_DIR` changes the parent directory rather than the timestamped run
directory. `run_config.txt` records the mode, input and output paths, MoE
runner, static-memory fraction, and decode CUDA graph limit used for the run.

## Monitor and inspect

While the job is active:

```bash
scripts/deepseek_v4_csa_topk/status.sh \
  results/deepseek_v4_csa_topk/latest --watch
tail -F results/deepseek_v4_csa_topk/latest/k*/server_*.err
tail -F results/deepseek_v4_csa_topk/latest/k*/client_*.err
tail -F results/deepseek_v4_csa_topk/latest/k*/server_*.log
```

`status.sh` combines four signals: the saved experiment phase/server PID, the
HTTP health endpoint, the age/size of the newest log or trace, and per-GPU
utilization/memory. By default it reports `STUCK-SUSPECTED` (exit code 2) only
when health is unavailable, no observed file has changed for 1200 seconds, and
all GPUs are below 10% utilization. Override these conservative thresholds with
`STALL_THRESHOLD_SEC` and `GPU_BUSY_THRESHOLD`; use `WATCH_INTERVAL_SEC` to
change the 30-second watch interval. Run it on the allocated GPU node so that
the recorded PID and `nvidia-smi` refer to the correct host.

The log stopping near `SymmDeviceMemory` does not by itself prove a hang. The
first 0731/FlashInfer startup can spend 10–15 minutes compiling/autotuning and
capturing CUDA graphs without frequent log lines. If GPU utilization remains
nonzero, memory is allocated on all TP ranks, and the server process is alive,
continue waiting. Treat it as likely stuck when all four monitor signals remain
negative beyond the threshold; then inspect every `server_*.err`, not only TP0.

After it exits:

```bash
cat results/deepseek_v4_csa_topk/latest/REPORT.md
column -s, -t results/deepseek_v4_csa_topk/latest/summary.csv
column -s, -t results/deepseek_v4_csa_topk/latest/h2d_tokens_by_accepted_tokens.csv | less
find -L results/deepseek_v4_csa_topk/latest -name '*.err' -size +0 -print
```

### Generate a partial report after a failed K group

The analyzer skips incomplete groups and records them under `Incomplete groups`
in `REPORT.md`. For example, if K=4096 exits because there is not enough free
GPU memory but K=512/1024/2048 completed, run:

```bash
python3 scripts/deepseek_v4_csa_topk/analyze.py \
  --results-dir results/deepseek_v4_csa_topk/latest
```

It discovers the three complete groups, emits their summary/curve, and marks
K=4096 as missing. To explicitly analyze only the successful groups, use
`--ks 512 1024 2048`. A group is complete only when both `benchmark.jsonl` and
`h2d_trace.tp0.jsonl` exist and the trace contains at least one record.

The SVG beside the CSV files is the requested mean per-request H2D-token curve.
Each `k*/server_*.err` contains startup/runtime exceptions, each
`k*/client_*.err` contains workload failures, and the corresponding `.log`
files retain ordinary progress output. A nonzero process exit stops the sweep,
so the first incomplete K identifies the failing group.
