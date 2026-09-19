# DSpark verification-step CSA Top-K similarity experiment

This experiment extends `dsv4_kv_residency` and measures Jaccard overlap
(`intersection / union`) between every pair of draft-token CSA Top-K sets in a
verification step. It also reports the subset formed by adjacent draft tokens.
Counters are accumulated across CSA layers before the ratio is calculated.

The gpuq entry point accepts the same environment variable names as the KV
residency job (`MODEL_PATH`, `RANDOM_INPUT_LEN`, `RANDOM_OUTPUT_LEN`, and
`NUM_PROMPTS`). Each run is written to its own UTC timestamp directory under
`${RESULTS_DIR:-results}/dspark_verfication_step_similarity/`, for example
`results/dspark_verfication_step_similarity/20260918_143015/`.
Set `RUN_TIMESTAMP` explicitly when a stable run identifier is needed.

The benchmark supports its standard datasets, including LongBench-v2. Use
`DATASET_NAME=longbench_v2` (the `longbench-v2` alias is also accepted), point
`DATASET_PATH` at a local JSONL or Parquet file, and optionally set
`LONGBENCH_CONTEXT_LEN` to discard examples that do not fit the configured
context window. `RANDOM_OUTPUT_LEN` fixes the LongBench-v2 response length as
well as the random workload response length. CUDA graphs are disabled because
this diagnostic probe performs Python-side set comparisons.

Example:

```bash
/usr/local/bin/gpuq run \
  --user td69032 \
  --project dspark_h2d_latency \
  --gpus 4 \
  --timeout 86400 \
  --env PORT=33464 \
  --env "PYTHONPATH=$PWD/python" \
  --env "PATH=/home/jovyan/td69032/lagom/.venv/bin:$PATH" \
  --env DETERMINISTIC_INFERENCE=0 \
  --env MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/ \
  --env DATASET_NAME=longbench_v2 \
  --env DATASET_PATH=/home/jovyan/td69032/LongBench-v2/data.parquet \
  --env LONGBENCH_CONTEXT_LEN=131072 \
  --env RANDOM_INPUT_LEN=128000 \
  --env RANDOM_OUTPUT_LEN=512 \
  --env NUM_PROMPTS=100 \
  --env TP_SIZE=4 \
  --env MEM_FRACTION_STATIC=0.80 \
  --env SERVER_RESTART_DELAY=5 \
  --env NCCL_SOCKET_IFNAME=lo \
  -- bash scripts/dspark_verfication_step_similarity/gpuq_entry.sh
```

`gpuq_entry.sh` starts the instrumented server, waits for it to become healthy,
forces a full DSpark verification budget, runs the configured workload,
dumps the records, and executes the analyzer. `RANDOM_INPUT_LEN=128000` and
`RANDOM_OUTPUT_LEN=512` are passed directly to `sglang.benchmark.serving`. For
LongBench-v2, `RANDOM_INPUT_LEN` is only recorded in the diagnostic metadata;
the actual input length comes from each dataset example and is bounded by
`LONGBENCH_CONTEXT_LEN` when it is set.

Run the command above from the repository root. For convenience, it is also
available without the surrounding explanation in `launch_command.sh`:

```bash
bash scripts/dspark_verfication_step_similarity/launch_command.sh
```

The spelling `verfication` in the directory and output name is intentional and
matches the requested experiment name.
