# DeepSeek V4 CSA Top-K experiment output

This directory is populated by `scripts/deepseek_v4_csa_topk/gpuq_entry.sh`.
No measurements are checked in because this checkout has no GPU. The completed job writes
`summary.csv`, `h2d_tokens_by_step.csv`, `h2d_tokens_by_decode_step.svg`, and `REPORT.md` here.

Inspect live errors with `tail -f results/deepseek_v4_csa_topk/k*/server_*.err` and
`tail -f results/deepseek_v4_csa_topk/k*/client_*.err`. After completion, read `REPORT.md`
or `column -s, -t results/deepseek_v4_csa_topk/summary.csv`.

The performance and trace passes are intentionally separate: the trace probe synchronizes the
GPU to obtain cache-miss copy latency and counts, so its TPOT is not reported. Each K uses the
same deterministic first 100 ShareGPT rows and serial request concurrency.
