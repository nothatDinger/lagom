# DSpark + HiSparse CSA KV residency experiment

1. Copy `config.env.example` to `config.env` and set the target model and dataset paths.
2. Submit `entrypoint.sh config.env` as the gpuq job command. The job has no script-level timeout.
3. Follow errors/progress with `tail -F "${RESULTS_DIR:-results}/dsv4_kv_residency/"{server,client}.log`.
4. After completion, read `${RESULTS_DIR:-results}/dsv4_kv_residency/report.md` and inspect the PNG/CSV/raw JSON.

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
