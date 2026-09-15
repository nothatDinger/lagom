# DeepSeek V4 HiSparse + DSpark CSA Top-K experiment

## Design

The controlled variable is CSA `top_k`: 512, 1024, 2048, and 4096. Every group
uses the deterministic first `NUM_PROMPTS` records from the ShareGPT JSON array
(100 by default), seed 0, and concurrency 1. SGLang's deterministic inference
mode is controlled by `DETERMINISTIC_INFERENCE` (enabled by default). The DSpark verify width is
`DSPARK_BLOCK_SIZE + 1` (the 0731 checkpoint default is 5 + 1); consequently the resident HiSparse buffer is set to
`verify_width * K`, the minimum safe size for a verify window whose Top-K sets
are disjoint.

The deterministic-inference setting is applied consistently to both the
performance and trace launches, so all four K groups in one run use the same
execution mode. Set `DETERMINISTIC_INFERENCE=0` for a non-deterministic control
run; do not add the corresponding CLI flag through `SERVER_EXTRA_ARGS`.
The server always receives `--disable-radix-cache`, which is a mandatory
HiSparse constraint; omitting it causes SGLang argument validation to fail
before the model workers start.

DeepSeek-V4-Flash-0731 contains FP4 MoE weights. The experiment therefore uses
`MOE_RUNNER_BACKEND=flashinfer_mxfp4` by default, matching the official 0731
launch recipe. Letting this checkpoint fall back to the Triton MoE runner can
fail during CUDA graph capture with `AssertionError: Hidden size mismatch`.
Override the variable only when the selected checkpoint and installed backend
are known to use a different compatible weight layout.

### Deterministic mode and the MXFP4 runner

`--enable-deterministic-inference` and
`--moe-runner-backend=flashinfer_mxfp4` are compatible configuration options in
this SGLang tree: deterministic-mode argument resolution changes the sampling,
attention, and collective choices but does not replace or reject the explicitly
selected MoE runner. The MXFP4 implementation dispatches separately on SM90,
SM100, and SM120. On SM90 it requires FlashInfer with the mixed-input MXFP4
helpers (FlashInfer PR #3084, version 0.6.11 or newer); a missing helper produces
an explicit startup error rather than silently falling back to Triton.

Here, "compatible" means the server supports and can launch the combination; it
does not mean results from different GPU architectures or different FlashInfer
versions are bitwise identical. Compare deterministic and non-deterministic runs
only on the same node, container, checkpoint, and backend version. Check
`run_config.txt` and the server log before using the measurements.

There are two server launches per K:

1. **perf** uses normal CUDA graphs and supplies the reported mean TPOT from
   `sglang.benchmark.serving`.
2. **trace** disables CUDA graphs and enables the intrusive H2D probe. The probe
   times each layer's cache-miss copy with CUDA events and records actual miss
   entries per request. Its synchronized TPOT is deliberately discarded.

Mean H2D latency is the mean, over decode steps, of the sum of all layer copy
times. `H2D / TPOT` uses milliseconds divided by milliseconds. The transfer
curve reports layer-0 cache-miss entries (logical C4 cache tokens) per request;
this avoids incorrectly multiplying logical token volume by the layer count.

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
`YYYYmmddTHHMMSSZ_det_on` or `YYYYmmddTHHMMSSZ_det_off` under
`results/deepseek_v4_csa_topk`. The `latest` symlink points to the newest run.
If two jobs use the same timestamp and mode, a numeric suffix prevents overwrite.
`RUN_TIMESTAMP` can be supplied by a job scheduler to override the UTC timestamp,
and `RESULTS_DIR` changes the parent directory rather than the individual run
directory. `run_config.txt` records the mode, input paths, and MoE runner used
for the run.

## Monitor and inspect

While the job is active:

```bash
tail -F results/deepseek_v4_csa_topk/latest/k*/server_*.err
tail -F results/deepseek_v4_csa_topk/latest/k*/client_*.err
tail -F results/deepseek_v4_csa_topk/latest/k*/server_*.log
```

After it exits:

```bash
cat results/deepseek_v4_csa_topk/latest/REPORT.md
column -s, -t results/deepseek_v4_csa_topk/latest/summary.csv
column -s, -t results/deepseek_v4_csa_topk/latest/h2d_tokens_by_step.csv | less
find -L results/deepseek_v4_csa_topk/latest -name '*.err' -size +0 -print
```

The SVG beside the CSV files is the requested mean per-request H2D-token curve.
Each `k*/server_*.err` contains startup/runtime exceptions, each
`k*/client_*.err` contains workload failures, and the corresponding `.log`
files retain ordinary progress output. A nonzero process exit stops the sweep,
so the first incomplete K identifies the failing group.
