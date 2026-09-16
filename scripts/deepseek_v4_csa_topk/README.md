# DeepSeek V4 HiSparse + DSpark CSA Top-K experiment

## Design

The controlled variable is CSA `top_k`: 512, 1024, 2048, and 4096. Every group
uses the deterministic first `NUM_PROMPTS` records from the ShareGPT JSON array
(100 by default), seed 0, and concurrency 1. The DSpark verify width is
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
times. `H2D / TPOT` uses milliseconds divided by milliseconds. The transfer
curve reports the physical copies summed across C4 layers in token-layer entries;
the CSV also reports the per-layer mean in logical C4 tokens.

If every `h2d_tokens_per_request` value is zero, first check prompt geometry.
The HiSparse kernel deliberately takes a zero-copy fast path whenever the
compressed C4 sequence length is no larger than `device_buffer_size`. With
DSpark this experiment sizes the buffer to `(DSPARK_BLOCK_SIZE + 1) * K`, and
one C4 entry represents four original tokens. With the default block size 5,
cache misses therefore require sequences longer than roughly `24 * K` original
tokens: 12,288 for K=512, 24,576 for K=1024, 49,152 for K=2048, and 98,304 for
K=4096. Ordinary short ShareGPT conversations can legitimately produce all
zeros. Use a long-context corpus (while keeping the same deterministic sampling
rule for every K) when the objective is to exercise H2D misses.

The analyzer now sums miss counts from every layer rather than reading layer 0
only. It adds a `Sampling diagnostics` warning to `REPORT.md` when all measured
requests fit in the resident buffer. New traces additionally record
`device_buffer_size` and compressed sequence lengths for direct verification.

## Run through gpuq

From the repository root, copy or source `env.example`, set both required
path variables, and submit the single entry point with your site's gpuq syntax:

```bash
source scripts/deepseek_v4_csa_topk/env.example
# Example only; gpuq flags differ by cluster:
gpuq scripts/deepseek_v4_csa_topk/gpuq_entry.sh
```

`MODEL_PATH` and `DATASET_PATH` are mandatory environment variables. `MODEL_PATH`
must point to DeepSeek-V4-Flash-0731, whose bundled DSpark draft head is loaded
from the same checkpoint; do not set `--speculative-draft-model-path`.
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
column -s, -t results/deepseek_v4_csa_topk/latest/h2d_tokens_by_step.csv | less
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
