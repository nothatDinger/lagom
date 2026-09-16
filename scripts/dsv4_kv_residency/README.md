# DSpark + HiSparse CSA KV residency experiment

1. Copy `config.env.example` to `config.env` and set the model and dataset paths for a local run.
2. Use `entrypoint.sh config.env` locally, or use the environment-only `gpuq_entry.sh` shown below.
3. Follow errors/progress with `tail -F "${RESULTS_DIR:-results}/dsv4_kv_residency/"{server,client}.log`.
4. After completion, read `${RESULTS_DIR:-results}/dsv4_kv_residency/report.md` and inspect the PNG/CSV/raw JSON.

The default deterministic ShareGPT policy selects the first 100 valid human prompts. Set
`DATASET_SAMPLING=random` and `RANDOM_SEED` for seeded random sampling. The run enables the
full DSpark debug record and pre-swap residency probe; use a full verify budget so every
proposal has a hypothetical CSA Top-K observation. Each hit ratio aggregates the HBM hits
and Top-K denominator over all observed CSA layers before any swap-in for that layer.
DeepSeek-V4-Flash-0731 bundles the DSpark draft head, so no separate draft-model path is
accepted or needed. The launcher explicitly disables radix caching and rejects HiCache
flags in `SERVER_EXTRA_ARGS`. It defaults to the checkpoint-compatible
`flashinfer_mxfp4` MoE runner; override `MOE_RUNNER_BACKEND` only for a compatible
checkpoint/hardware combination. `RESULTS_DIR` selects the base output directory and
defaults to the repository's `results` directory.

CUDA graphs are disabled because the probe intentionally executes Python-side tensor
bookkeeping on every verify step; this experiment measures residency, not throughput.

## gpuq launch

Run this command from the repository root:

```bash
gpuq run \
  --user td69032 \
  --project dspark_filter_experiment \
  --gpus 4 \
  --env "PYTHONPATH=$PWD/python" \
  --env "PATH=$PWD/.venv/bin:$PATH" \
  --env DETERMINISTIC_INFERENCE=0 \
  --env MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/ \
  --env DATASET_PATH=/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json \
  --env DATASET_NAME=sharegpt \
  --env DATASET_SAMPLING=head \
  --env DATASET_SAMPLE_SIZE=10 \
  --env NUM_PROMPTS=10 \
  --env TP_SIZE=4 \
  --env SERVER_READY_TIMEOUT_SECONDS=0 \
  --env SERVER_STOP_TIMEOUT_SECONDS=0 \
  --env MEM_FRACTION_STATIC=0.78 \
  -- bash scripts/dsv4_kv_residency/gpuq_entry.sh
```

`DATASET_SAMPLING=head` deterministically selects the first
`DATASET_SAMPLE_SIZE` valid ShareGPT prompts. `random` enables seeded sampling.
`SERVER_READY_TIMEOUT_SECONDS=0` means wait indefinitely for model startup.
`SERVER_STOP_TIMEOUT_SECONDS=0` means wait indefinitely for graceful server shutdown.
`NUM_PROMPTS` is accepted as a fallback when `DATASET_SAMPLE_SIZE` is omitted.
