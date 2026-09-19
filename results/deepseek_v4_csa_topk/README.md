# DeepSeek V4 CSA Top-K experiment output

This directory is populated by `scripts/deepseek_v4_csa_topk/gpuq_entry.sh`.
Each invocation creates a timestamped `YYYYmmddTHHMMSSZ_det_off` child directory
and updates the `latest` symlink.
Timestamp collisions receive a numeric suffix instead of overwriting an older run.
No measurements are checked in because this checkout has no GPU. The completed job writes
`summary.csv`, `h2d_tokens_by_accepted_tokens.csv`,
`h2d_tokens_by_accepted_tokens.svg`, and `REPORT.md` in that child directory.
Each K directory retains `benchmark.jsonl` from the perf pass and
`benchmark_trace.jsonl` from the intrusive trace pass; only perf TPOT is reported.
If a larger K fails, rerun `analyze.py --results-dir .../latest`; incomplete K
groups are skipped and listed in the generated report instead of aborting analysis.
The H2D curve is expressed in token-layer entries summed over C4 layers, with a
per-layer logical-token mean in the CSV. `Sampling diagnostics` explains an
all-zero result when every request fits the resident C4 device buffer.

Inspect live errors with `tail -f results/deepseek_v4_csa_topk/latest/k*/server_*.err` and
`tail -f results/deepseek_v4_csa_topk/latest/k*/client_*.err`. After completion, read
`latest/REPORT.md` or `column -s, -t results/deepseek_v4_csa_topk/latest/summary.csv`.
Run `scripts/deepseek_v4_csa_topk/status.sh .../latest --watch` on the GPU node
to combine process, health, file-progress, and GPU-utilization signals when a
startup log appears to stop advancing.

The performance and trace passes are intentionally separate: the trace probe synchronizes the
GPU to obtain cache-miss copy latency and counts, so its TPOT is not reported. Each K uses the
same deterministic first 100 ShareGPT rows and serial request concurrency.
Both passes run with deterministic inference disabled. DeepSeek V4 requires the
`dsv4` attention backend, which this SGLang version excludes from deterministic
inference; requesting `DETERMINISTIC_INFERENCE=1` fails fast in the entry script.
The DeepSeek-V4-Flash-0731 checkpoint supplies both target weights and its bundled
DSpark draft head, so the experiment does not require a separate draft-model path.
The launch command also disables the radix cache, as required by HiSparse.
For the FP4 0731 checkpoint it defaults to the `flashinfer_mxfp4` MoE runner;
`run_config.txt` records the selected runner for later diagnosis.
