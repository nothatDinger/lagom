#!/usr/bin/env bash
set -Eeuo pipefail

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
