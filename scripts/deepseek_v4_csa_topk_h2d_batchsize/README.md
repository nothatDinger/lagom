# DeepSeek V4 CSA H2D latency by request batch size

This experiment is derived from `deepseek_v4_csa_topk`, but fixes CSA `top_k` at
512 and sweeps concurrent request batch sizes `1 4 8 16 32`. For each batch size
it compares the mean H2D latency of a decode/verify cycle with the non-intrusive
performance pass's mean TPOT.

## Run

```bash
source scripts/deepseek_v4_csa_topk_h2d_batchsize/env.example
# Set MODEL_PATH and, on offline nodes, DATASET_PATH first.
gpuq scripts/deepseek_v4_csa_topk_h2d_batchsize/gpuq_entry.sh
```

`BATCH_SIZES` is a space-separated list and defaults to `"1 4 8 16 32"`.
K is fixed at `512`. The runner sets both `--num-prompts` and `--max-concurrency` to
the current batch size, so every group contains one simultaneous request batch.
The largest batch size determines how many ShareGPT/LongBench records are
prepared. Random data needs no preprocessing.

The default 110,000-token random request is inherited from the H2D latency test
because it exceeds the fixed K=512 resident C4 buffer and exercises actual H2D
copies. Ensure the target system has enough memory for the requested concurrent
long-context workload, or override the dataset/input length deliberately.

Every batch size has two isolated server runs:

1. `perf` records TPOT in `benchmark.jsonl` without the synchronizing trace probe.
2. `trace` disables CUDA graphs and records H2D events in
   `h2d_trace.tp0.jsonl`; trace-run TPOT is intentionally ignored.

Results are written under
`results/deepseek_v4_csa_topk_h2d_batchsize/<timestamp>_det_off/`, with
subdirectories `bs1`, `bs4`, `bs8`, `bs16`, and `bs32`. The runner updates the
`latest` symlink and generates:

- `summary.csv`: batch size, mean H2D, mean TPOT, and H2D/TPOT ratio;
- `h2d_tokens_by_accepted_tokens.csv` and `.svg`: cache-miss diagnostics;
- `REPORT.md`: a readable summary and diagnostics.

## Analyze or monitor

```bash
python3 scripts/deepseek_v4_csa_topk_h2d_batchsize/analyze.py \
  --results-dir results/deepseek_v4_csa_topk_h2d_batchsize/latest

# Analyze a subset:
python3 scripts/deepseek_v4_csa_topk_h2d_batchsize/analyze.py \
  --results-dir /path/to/run --batch-sizes 1 4 8

scripts/deepseek_v4_csa_topk_h2d_batchsize/status.sh \
  results/deepseek_v4_csa_topk_h2d_batchsize/latest --watch
```

Incomplete groups are skipped and listed in `REPORT.md`. A group is complete
when both `bsN/benchmark.jsonl` and `bsN/h2d_trace.tp0.jsonl` exist.
