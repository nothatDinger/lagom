# DeepSeek V4 HiSparse + DSpark CSA Top-K experiment

## Design

The controlled variable is CSA `top_k`: 512, 1024, 2048, and 4096. Every group
uses the deterministic first `NUM_PROMPTS` records from the ShareGPT JSON array
(100 by default), seed 0, concurrency 1, and SGLang's
`--enable-deterministic-inference` mode. The DSpark verify width is
`DSPARK_BLOCK_SIZE + 1` (the 0731 checkpoint default is 5 + 1); consequently the resident HiSparse buffer is set to
`verify_width * K`, the minimum safe size for a verify window whose Top-K sets
are disjoint.

Deterministic inference is enabled for both the performance and trace launches,
so batch-invariant kernels and deterministic sampling remain identical across
all four K groups. Do not override this flag through `SERVER_EXTRA_ARGS`.

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

From the repository root, copy or source `env.example`, set all three required
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

## Monitor and inspect

While the job is active:

```bash
tail -F results/deepseek_v4_csa_topk/k*/server_*.err
tail -F results/deepseek_v4_csa_topk/k*/client_*.err
tail -F results/deepseek_v4_csa_topk/k*/server_*.log
```

After it exits:

```bash
cat results/deepseek_v4_csa_topk/REPORT.md
column -s, -t results/deepseek_v4_csa_topk/summary.csv
column -s, -t results/deepseek_v4_csa_topk/h2d_tokens_by_step.csv | less
find results/deepseek_v4_csa_topk -name '*.err' -size +0 -print
```

The SVG beside the CSV files is the requested mean per-request H2D-token curve.
Each `k*/server_*.err` contains startup/runtime exceptions, each
`k*/client_*.err` contains workload failures, and the corresponding `.log`
files retain ordinary progress output. A nonzero process exit stops the sweep,
so the first incomplete K identifies the failing group.
