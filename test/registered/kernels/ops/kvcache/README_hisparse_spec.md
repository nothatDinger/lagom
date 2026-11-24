# DeepSeek-V4 / DSPARK KV verification

The C4 resolver on `perf/swapin_oneshot` follows the target KV rules in sections 3.1–3.4 of *Adaptive DHiSparse*. The implementation builds on the standalone fused swap-in in `swapin_oneshot@8948c57fb` and adapts the KV code in `DHisparse@1ad6ad8db` to the existing DSPARK, PD and H2D measurement paths. [Paper](https://arxiv.org/html/2610.13708v1#S3)

| Mechanism | Original `swapin_oneshot` | Current CUDA V4 / DSPARK path |
|---|---|---|
| Persistent hot capacity `B` | Default `2 × verify_width × top_k`. The whole union had to fit in hot and within 8192 entries. | Explicit page-aligned `B` survives width changes. The default of `16 × model.index_topk` adopts the paper's V4.1 ratio as a starting point for this model. |
| Verification arena | The extra page shared space with the frontier and rotating scratch. | An owned payload slab has a separate, page-rounded slice of `Kmax × top_k` rows for each active batch row and simultaneously live source. Sequential V4 C4 attention layers share one physical arena. PD transfer slots have persistent request metadata and consume arena space only while active. |
| Resolution and promotion | The fused gather/commit pair could rotate physical hot slots before attention. | Resolve tail and ready optional history first, then hot and host. Deduplicate misses across valid queries by first occurrence and preserve selection order. Attention reads the resulting addresses. After that read, unselected LRU victims receive arena-to-hot copies at stable addresses. Overflow expires when the next source reuses the arena. |
| Import and pressure | PD imports C4 to pinned host. Local prefill eagerly backs up its history. | Retain complete local prefill pages while capacity allows and back up the partial boundary. PD keeps its Mooncake host receive, and can populate complete spare GPU pages at admission. Publish copies after completion. Whole-request reclaim waits for backup and readers, preserves the reserve, and bounded round-robin refill repopulates backed history. |
| Provisional writes | A separate fixed writer scratch and mapping covered the current DSPARK block. | A protected per-request tail holds the frontier and provisional C4 rows. Query visibility uses the actual number of completed compressed rows, including zero. An accepted prefix commits exactly the C4 boundaries in `[floor(prefix/4), floor((prefix+commit_len)/4))`. The next forward computes the bonus token's KV. |
| Force-H2D control | The performance branch added an optional reread of resident hits to the original fused kernel. | Both arms use the same resolver, attention addresses, LRU and admissions. The forced arm rereads each eligible host-backed hot hit once into an unused occurrence in the arena. The trace counts native H2D, additional H2D and D2D admissions separately. |

The adaptive coordinator calls `resolve` (plan and native refill), optionally `reread`, then `promote` (victim selection and D2D copy) after attention. The adaptive baseline has only timing markers around the reread slot. The resolver counts eligible rows before attention and the copy kernel independently counts completed rereads. Byte-level tests check the arena destinations, attention addresses and preserved hot rows. The legacy `load_cache_to_device_buffer_spec_mla` path, selected with `"adaptive": false`, launches a scan in both traced arms and writes forced rereads to the existing hot slots.

For example, the paired PD run uses `Kmax=6`, `k=512` and `B=2048`. At 256 active requests, the single live C4 arena holds 786,432 rows per decode rank. That is 438 MiB of KV payload, or 438.75 MiB with the 64-row packed page pitch. Hot, tail and metadata have their own budgets. The original hash/CLOCK path is available with `"adaptive": false`. [PD configuration and trace interpretation](../../../../../benchmark/lagom/README.md)

## Model and transport differences

| Implementation | KV source and draft layout |
|---|---|
| Paper, DeepSeek-V4.1-Flash / DSpark | Eight selecting layers read four physical KV sources. Reuse consumers inherit the materialized selection, and Reindex consumers resolve new selections against the shared source. A missed row carries 288 bytes. The evaluated hot budget is `16 × 512 = 8192` rows. |
| This branch, DeepSeek-V4 / DSPARK | Each C4 layer owns its payload and hot state. The packed row carries 576 value/RoPE bytes and 8 scale bytes. Each source finishes its attention and promotion before the next C4 layer resolves. DSPARK uses its existing SWA draft storage and PD transfer layout. |
| Paper, GLM-5.3 / DFlash2 | 21 index producers supply selections to 78 distinct payload layers. Overlapping follower reads require separate arenas. The evaluated budget is `2 × 2048 = 4096` rows. With window 2048, page 64 and maximum verification width 8, the separate draft ring holds 2176 rows per request under equation 9. |

Optional history in this branch borrows dedicated C4 payload pages. The V4 index pool uses a different row format and its logical pages are registered independently with Mooncake. Lending index backing would require coordinating those allocations and resolving the resulting payload addresses. GPU-first RDMA placement would also require per-request receive destinations. These differences affect residency and transfer cost when comparing to the paper. [Paper architecture](https://arxiv.org/html/2610.13708v1#S3) · [Paper cache-budget comparison](https://arxiv.org/html/2610.13708v1#S4.SS3)

## Tests

From the repository root, with the project uv environment active:

```bash
uv run --active --no-project python -m pytest -q \
  test/registered/unit/mem_cache/test_dhisparse.py \
  test/registered/unit/managers/test_hisparse_dspark.py \
  test/registered/unit/mem_cache/test_hisparse_spec_swapin.py \
  benchmark/lagom
```

On NVIDIA hardware, add `test/registered/kernels/ops/kvcache/test_dhisparse_cuda.py` and `test/registered/kernels/ops/kvcache/test_hisparse_spec.py`. The adaptive GPU suite checks packed values and scales in eager execution and CUDA Graph replay with reordered requests, padding, warm hits and union overflow.
