# DeepSeek V4 CSA Top-K experiment output

This directory is populated by `scripts/deepseek_v4_csa_topk/gpuq_entry.sh`.
Each invocation creates a timestamped `YYYYmmddTHHMMSSZ_det_on` or
`YYYYmmddTHHMMSSZ_det_off` child directory and updates the `latest` symlink.
Timestamp collisions receive a numeric suffix instead of overwriting an older run.
No measurements are checked in because this checkout has no GPU. The completed job writes
`summary.csv`, `h2d_tokens_by_step.csv`, `h2d_tokens_by_decode_step.svg`, and `REPORT.md` in that child directory.

Inspect live errors with `tail -f results/deepseek_v4_csa_topk/latest/k*/server_*.err` and
`tail -f results/deepseek_v4_csa_topk/latest/k*/client_*.err`. After completion, read
`latest/REPORT.md` or `column -s, -t results/deepseek_v4_csa_topk/latest/summary.csv`.

The performance and trace passes are intentionally separate: the trace probe synchronizes the
GPU to obtain cache-miss copy latency and counts, so its TPOT is not reported. Each K uses the
same deterministic first 100 ShareGPT rows and serial request concurrency.
Both passes use the same `DETERMINISTIC_INFERENCE` setting. The `det_on`/`det_off`
directory suffix and `run_config.txt` make that setting explicit.
The DeepSeek-V4-Flash-0731 checkpoint supplies both target weights and its bundled
DSpark draft head, so the experiment does not require a separate draft-model path.
The launch command also disables the radix cache, as required by HiSparse.
For the FP4 0731 checkpoint it defaults to the `flashinfer_mxfp4` MoE runner;
`run_config.txt` records the selected runner for later diagnosis.
This runner can be selected together with deterministic inference, but comparisons
must keep the GPU architecture and FlashInfer/SGLang versions fixed.
