# 满 running batch 的 decode 性能测试

入口 `benchmark/lagom/run_concurrent.py`，在 `perf/lagom` 分支运行。
`--max-concurrency` 可选 **8 / 16 / 32 / 64**，每次选择一个值，默认 8。
比较 `lagom`、`hisparse`（DSPARK+HiSparse）、`dspark` 三组。

默认沿用 packed ShareGPT **input=32000、output=1024、W=P=7、hot=8192**，
PD 融合、单机 TP。每组先跑一个输出 32 token 的预热请求，然后同时提交 C 个
不同 prompt；三个模式复用相同 input_ids 和 request ID。每组只提交这一批 C 个
正式请求，避免为维持压力不断补入新 prefill。GPU 是否有容量容纳 32K×C 的 KV，
仍需实际验证，脚本不会静默降低并发或输入长度。

## H100 运行示例

对于原生 MXFP4 target/draft，两边使用 Marlin：

```bash
python benchmark/lagom/run_concurrent.py \
  --model /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --draft /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --sharegpt /data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tp 4 --max-concurrency 16 \
  --moe-runner-backend marlin --speculative-moe-runner-backend marlin \
  --output ./results/lagom-c16
```

输出目录必须不存在。更换 `--max-concurrency 8/32/64` 时使用各自的新输出目录。
若 draft 权重实际上是 FP8，则单独设置 draft 后端为 `auto`；相关量化说明见
[单并发实验指导](README.md)。

默认 `--phase throughput`，三组各启动一次服务。CUDA Graph 只捕获 `[1,C]` 两个
batch size，减少启动时间；小于 C 的请求数即使 padding 到 C，也不会计入满 batch
性能。`--dry-run` 可以先检查命令，`--modes` 可限制要运行的组。

## 满 batch 如何判定

服务端 DSPARK worker 在每次 forward 前检查：

1. 本轮是纯 decode，不含 extend/mixed prefill/idle。
2. **真实 `len(batch.reqs)==C`**，所有请求属于本次正式测试。
3. 请求集合保持一致，且距离 `max_new_tokens` 仍有余量。考虑 overlap scheduler
   的 CPU output_ids 可能落后一轮，保留两个完整 verify window 的余量，避免把
   输出上限处丢弃的 token 计入吞吐。请求同时设置 `ignore_eos=true`。
4. 连续满 batch 预热 **4 轮**，再连续测量 **32 轮**。

未填满、prefill 插入、请求集合变化或收尾阶段变小，都会使未完成的窗口作废，重新
等待满 batch。每个 TP rank 必须记录同一组 forward 和接受 token 数；draft/target
出现 eager fallback 时本次测量报错。没有足够的连续满 batch 样本时，不输出有效
吞吐结果；检查 `decode/*.status.json` 和 `server.log`，确认实际最大 batch、KV
容量、retraction 和请求是否过早结束。

可以用 `--measure-steps 16 --full-batch-warmup 2` 缩短采样窗口。默认仍让正式请求
生成 1024 token；`--output-len` 可调整，但必须留出采样和收尾余量。默认每个请求
上限 600 秒、每组启动后预算 900 秒、服务启动预算 900 秒，可用相应 timeout 参数
调整。高并发的长输入 prefill 和模型加载可能耗时较长，不计入 decode 性能。

## 性能指标

性能阶段关闭逐层 trace，只在 worker forward 外围记录 CUDA events，兼容 graph
replay。接受 token 数和必要的长度快照保留在 GPU，完成采样后再读回，采样轮内不为
统计增加 D2H 或 synchronize。CUDA events 和快照仍有少量测量开销。

| 字段 | 定义 |
|---|---|
| `decode_tokens_per_s` | 满 batch 窗口内接受的输出 token / 窗口时长 |
| `window_ms_max_rank` | 从第一轮开始到最后一轮结束，取 TP 各 rank 的最大时长；包含轮间 scheduler/CPU 提交空洞 |
| `gpu_forward_tokens_per_s` | 相同 token 数 / 各轮 GPU forward 跨度之和，不包含轮间空洞 |
| `mean_per_request_tokens_per_s` | 满 batch decode 吞吐 / C |
| `mean_step_cycle_ms` | 完整窗口时长 / 采样轮数，包含轮间开销 |
| `mean_gpu_step_ms`、`p50_gpu_step_ms`、`p95_gpu_step_ms` | 各轮 GPU forward 跨度统计，每轮先取 TP rank 最大值 |
| `mean_accepted_tokens_per_request_per_step` | 每请求、每 verify 轮的平均接受长度，包含 bonus token |

token 数使用实际接受长度，**不使用 C×W 推算，也不在 TP rank 间重复累加**。
初始 prefill 生成的 token 不计入。这里是服务端 decode 性能，不包括 HTTP/SSE 传输、
TTFT、ramp-up 和完整请求端到端时间。CPU submit span、GPU step span 与轮间 gap
不是可直接相加的互斥分解。

输出：

- `decode_throughput.csv`：三组满 batch decode 性能对比，每完成一组即保存。
- `throughput/<mode>/decode_steps.csv`：逐轮真实 batch、接受 token 数、GPU/CPU 时间和轮间 gap。
- `throughput/<mode>/decode/rank-*.json`：每个 rank 的原始窗口、请求 ID、长度和时间。
- `decode/rank-*.status.json`：采样前的诊断信息；成功与否以最终 rank report 为准。
- `requests.json`、`cohort.json`：客户端请求及包含 prefill 的整批耗时，不参与上述 decode 吞吐。
- `manifest.json`、`prompts.jsonl`、各组 `launch.json`、`server_info.json`、`server.log`。

## 满 batch 的逐层 trace

需要定位 attention、MLP、refill/prefetch 等瓶颈时，另建输出目录并将上述命令改为：

```bash
--phase trace --trace-modes lagom hisparse dspark --nsys \
--output ./results/lagom-c16-trace
```

trace 阶段仍提交 C 个 32000/1024 请求；只采集实际 batch=C 的 decode，跳过最初
2 个符合条件的 step 后记录 12 个。prefill/idle 和不足 C 的 decode 不写入采样槽位。
这些诊断 step 可以不连续，通过 `forward_iter` 检查间隔；严格连续窗口的吞吐只取
性能阶段的 CUDA event report。详细 trace 与 Nsight 会扰动耗时，不能替代性能阶段。
`--phase all` 会先做三组性能测量，再启动独立服务做 trace，默认 trace 也包含三组。

当前开发环境无 GPU，真实 H100 上的高并发容量、graph replay 和性能仍需实测。

```bash
python -m pytest -q test/registered/unit/observability/test_lagom_concurrent.py \
  test/registered/unit/observability/test_lagom_perf.py
```
