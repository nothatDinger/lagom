# DSpark filter experiment

Set `MODEL_PATH` and `DATASET_PATH` in the process that owns the GPU allocation.
All output defaults to `<repo>/results/dspark_filter_experiment`.

## Direct execution

```bash
export MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash-0731
export DATASET_PATH=/datasets/ShareGPT.json
export DATASET_SAMPLING=head
export DATASET_SAMPLE_SIZE=100
bash scripts/dspark_filter_experiment/run_all.sh
```

## GPUQ execution

Submit `gpuq_job.sh` as **one GPUQ job** so the profiling server, profiler,
experiment server, and benchmark clients share the same allocation. GPUQ syntax
varies by installation; append this payload to the command your cluster uses:

```bash
env MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash-0731 \
    DATASET_PATH=/datasets/ShareGPT.json \
    TP_SIZE=4 \
  bash scripts/dspark_filter_experiment/gpuq_job.sh
```

For example, if your GPUQ accepts a command after `--`, submit it as:

```bash
gpuq <site allocation flags> -- \
  env MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash-0731 \
      DATASET_PATH=/datasets/ShareGPT.json \
      TP_SIZE=4 \
    bash scripts/dspark_filter_experiment/gpuq_job.sh
```

If the queued shell must activate a virtual environment, pass the setup command:

```bash
EXPERIMENT_ENV_SETUP_SCRIPT=/path/to/activate-sglang.sh
```

Do not place model or dataset paths in `config.sh`; those two values deliberately
have no defaults. Other supported overrides are documented in `config.sh`.

`DATASET_SAMPLING=head` selects the first `DATASET_SAMPLE_SIZE` rows before any
experiment starts, so every repetition sees the same subset. `random` performs a
reproducible sample controlled by `DATASET_SAMPLE_SEED`; `all` copies every row.
The prepared subset is stored under `results/dspark_filter_experiment/datasets`.
For ShareGPT's first 100 source rows, use:

```bash
export DATASET_NAME=sharegpt
export DATASET_SAMPLING=head
export DATASET_SAMPLE_SIZE=100
export NUM_PROMPTS=100
```
