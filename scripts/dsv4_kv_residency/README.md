# DSpark + HiSparse CSA KV residency experiment

1. Copy `config.env.example` to `config.env` and set the target model and dataset paths.
2. Submit `entrypoint.sh config.env` as the gpuq job command. The job has no script-level timeout.
3. Follow errors/progress with `tail -F "${RESULTS_DIR:-results}/dsv4_kv_residency/"{server,client}.log`.
4. After completion, read `${RESULTS_DIR:-results}/dsv4_kv_residency/report.md` and inspect the PNG/CSV/raw JSON.

For the gpuq environment used by the other DSpark experiments, the experiment can be
launched directly from the repository root without creating `config.env`:

```bash
gpuq run \
    --user td69032 \
    --project dsv4_kv_residency_experiment \
    --gpus 4 \
    --env "PYTHONPATH=$PWD/python" \
    --env "PATH=$PWD/.venv/bin:$PATH" \
    --env TARGET_MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/ \
    --env DATASET_PATH=/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json \
    --env DATASET_NAME=sharegpt \
    --env SAMPLE_COUNT=100 \
    --env SAMPLE_METHOD=first \
    --env RANDOM_SEED=0 \
    --env MAX_NEW_TOKENS=256 \
    --env TP_SIZE=4 \
    --env MOE_RUNNER_BACKEND=flashinfer_mxfp4 \
    --env 'HISPARSE_CONFIG={"top_k":2048,"host_to_device_ratio":5}' \
    -- bash scripts/dsv4_kv_residency/run_experiment.sh
```

`--gpus` and `TP_SIZE` must match. Change `SAMPLE_COUNT` to `10` for the same
small smoke-test size as the example command; keep `100` for the intended experiment.

The collector supports ShareGPT, LongBench, and LongBench-v2. Select the input schema
with `DATASET_NAME=sharegpt`, `longbench`, or `longbench-v2`. A dataset path may be a
JSON/JSONL/Parquet file or a directory containing those files. For the local
LongBench-v2 dataset, replace the two dataset options above with:

```bash
    --env DATASET_PATH=/home/jovyan/td69032/LongBench-v2 \
    --env DATASET_NAME=longbench-v2 \
```

For a LongBench v1 checkout, point `DATASET_PATH` at its root or `data` directory and
use `DATASET_NAME=longbench`. Parquet input requires `pandas` and `pyarrow` in the job
environment.

The default deterministic ShareGPT policy selects the first 100 valid human prompts. Set
`SAMPLE_METHOD=random` and `RANDOM_SEED` for seeded random sampling. The run enables the
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
