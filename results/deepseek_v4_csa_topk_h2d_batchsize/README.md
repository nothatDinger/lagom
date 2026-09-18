# DeepSeek V4 CSA H2D batch-size experiment output

This directory is populated by
`scripts/deepseek_v4_csa_topk_h2d_batchsize/gpuq_entry.sh`. Each invocation
creates a timestamped child directory and updates `latest`; measurements are not
checked in because this checkout has no GPU.

The fixed default is K=512 and the request batch-size sweep is 1, 4, 8, 16, and
32. Read `latest/REPORT.md` or `latest/summary.csv` after completion. Per-batch
logs and traces live in `latest/bs*/`.
