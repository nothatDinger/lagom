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
directory. `run_config.txt` records the mode and input paths used for the run.

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
