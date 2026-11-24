# 并发上限下的 decode 性能测试

入口 `benchmark/lagom/run_concurrent.py`，在 `perf/lagom` 分支运行。
`--max-concurrency` 可选 **8 / 16 / 32 / 64**，默认 8，表示客户端和服务端允许的
**最大并发**。实际 running batch 可以是 1 到 C，且可以逐轮变化。设置 C=8、实际
最大 batch=7 是合法结果，不要求必须达到 8，也不要求同一批请求贯穿整个采样窗口。

`--requests` 独立控制**每组、每个阶段的正式请求总数 N**，默认 16；更换并发上限
不会自动更改 N。预热请求和最后用于导出的 flush 请求不计入 N。

默认 packed ShareGPT **input=32000、output=1024、W=7、P=4、hot=8192**，PD 融合、
单机 TP。每组先跑一个输出 32 token 的预热请求，然后提交最多 `min(N,C)` 个正式
请求。每完成一个请求就补发下一个，直到完成 N 个不同 prompt，末尾不足 C 个也
正常执行。服务端根据容量和调度情况决定实际 running batch。
默认 `--phase throughput`，比较 `lagom`、`hisparse`（DSPARK+HiSparse）、`dspark`
三组；`--modes` 可限制组别。

例如 `--requests 19 --max-concurrency 8` 会完成 19 条正式请求，最多同时在途 8 条；
`--requests 3 --max-concurrency 8` 也合法，只会提交 3 条。默认 `16/8` 用于短测；
提高并发上限时可显式增加 N，持续施压通常可从 `N=2C` 开始，并按实际耗时调整。

## 控制变量

三组复用相同 input_ids、request ID、请求总数、输入/输出长度、并发上限、TP、W、采样参数、
随机种子、prefill chunk、显存比例、后端参数和 CUDA Graph 配置。关闭 radix cache，
设置 `ignore_eos=true`。不同模式的实际接受率和 batch 分布作为测量结果记录。

| 设置 | Lagom | DSPARK+HiSparse | DSPARK |
|---|---|---|---|
| `device_buffer_size` | 同一个 `--hot` | **与 Lagom 相同** | 不适用 |
| `host_to_device_ratio` | 2 | 2 | 不适用 |
| `force_h2d_each_step` | false | false | 不适用 |
| 预测窗口 | P 个额外 token | 无 | 无 |
| 预取 | Lagom | 关闭 | 无 HiSparse |

例如 `--hot 8192` 会同时给两个 HiSparse 组配置 8192，不能因为关闭预取就把基线
单独降到 4096。脚本禁止通过额外 server arguments 覆盖实验控制的 HiSparse 参数。
`manifest.json` 的 `comparison_controls`、各组 `launch.json` 和汇总 CSV 都保存
容量等对照配置。纯 DSPARK 仍使用原生 CSA，没有 HiSparse device buffer。

CUDA Graph 捕获 `[1,2,4,...,C]`，三组一致。实际 batch 使用合适的捕获档位；统计
始终取真实 `len(batch.reqs)`，padding 不计入并发或接受 token 数。增加捕获档位
会比旧版只捕获 `[1,C]` 多一些启动时间，但避免所有中间 batch 都 padding 到 C。

## H100 运行示例

对于原生 MXFP4 target/draft，两边使用 Marlin：

```bash
python benchmark/lagom/run_concurrent.py \
  --model /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --draft /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --sharegpt /data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tp 4 --requests 16 --max-concurrency 8 --hot 8192 \
  --moe-runner-backend marlin --speculative-moe-runner-backend marlin \
  --output ./results/lagom-n16-c8
```

输出目录必须不存在。更换并发上限时使用新的输出目录；`--dry-run` 可先检查三组
启动命令、请求总数与相同的 buffer 配置。若 draft 权重实际上是 FP8，可将 draft 后端设置为
`auto`；量化说明见[单并发实验指导](README.md)。

## 采样规则

服务端 DSPARK worker 在每次 forward 前检查：

1. 本轮是纯 decode，不含 extend/mixed prefill/idle。
2. **1 ≤ 真实 `len(batch.reqs)` ≤ C**，所有请求属于本次正式测试。
3. 每个请求的剩余输出预算大于 `2*(W+1)`。这包含 overlap scheduler 的 CPU
   output_ids 可能落后一轮的余量，避免将输出上限处丢弃的 token 计入吞吐。
4. 连续合格 decode 预热 **4 轮**，再连续采样 **32 轮**。

在上述范围内，batch 大小或请求集合变化不会重置窗口。prefill 插入、非正式请求或
输出余量不足会使未完成窗口作废，重新预热。测量的是第一段符合条件的短 decode
窗口，不是所有请求生命周期的平均吞吐。只有合格连续轮数不足、TP rank 不一致或
draft/target graph fallback 等情况才报错；实际 batch 小于上限不会报错。

补发请求可能插入新的 prefill，尚未完成的采样窗口仍按上述规则重新开始。完成
32 轮采样后，会继续执行剩余正式请求，直到完成 N 条；不会把客户端 N 条请求的
完整耗时当作纯 decode 窗口吞吐。`--requests` 控制总负载，`--measure-steps` 控制
采样轮数，两者含义不同。增加 N 不会自动延长每组运行预算。

用 `--measure-steps 16 --decode-warmup 2` 可以缩短采样。
旧参数 `--full-batch-warmup` 保留为 `--decode-warmup` 的兼容别名，含义同样不要求
满 batch。默认每个请求上限 600 秒、每组启动后预算 900 秒、启动预算 900 秒。
加载、prefill 和未进入采样窗口的阶段不计入 decode 性能。

默认输出仍为 1024，无需为了达到并发上限增加到 2048。若要延长负载，可对三组
统一显式设置 `--output-len 2048`，并使用新目录；该配置会增加请求时间和 KV 需求，
应与 1024 的结果分开标注。输入长度、输出长度和采样配置写入 manifest 和汇总。

## 性能指标

性能阶段关闭逐层 trace，在 worker forward 外围记录 CUDA events，兼容 graph
replay。接受 token 数和必要的长度快照完成采样后再读回，采样轮内不为统计增加
D2H 或 synchronize。CUDA events、快照及 CPU 诊断计数仍有少量测量开销。

| 字段 | 定义 |
|---|---|
| `requests` | 每组、每阶段完成的正式请求总数 N |
| `max_concurrency` | 配置的并发上限 C |
| `client_concurrency_limit` | 客户端在途请求上限 `min(N,C)` |
| `min/mean/max_running_batch_size` | 采样轮实际 batch 的最小值、按轮平均值和最大值 |
| `running_batch_histogram` | 每种实际 batch 的采样轮数 |
| `decode_tokens_per_s` | 窗口内实际接受的输出 token / 窗口时长 |
| `window_ms_max_rank` | 首轮开始至末轮结束，取 TP rank 最大时长，包含轮间 scheduler/CPU 空洞 |
| `gpu_forward_tokens_per_s` | 相同 token 数 / 各轮 GPU forward 跨度之和，排除轮间空洞 |
| `mean_step_cycle_ms` | 完整窗口时长 / 采样轮数 |
| `mean/p50/p95_gpu_step_ms` | GPU forward 跨度统计，每轮先取 TP rank 最大值 |
| `mean_accepted_tokens_per_request_per_step` | 总接受 token / 各轮真实 batch 之和 |

接受长度允许 **1 ≤ accepted_tokens ≤ W+1**，包含 bonus token。token 数不使用
C×W 推算，不在 TP rank 间重复累加，初始 prefill token 不计入。旧的
`mean_per_request_tokens_per_s=吞吐/C` 已移除，实际 batch 可变时不能按上限均分。

若三组实际 batch 分布不同，应同时查看 `decode_by_batch_size.csv` 中相同 batch
的 GPU forward 耗时、吞吐和接受长度；没有相同 batch 的样本时，不宣称同 batch
加速比。分桶指标只累加对应 forward 的 GPU 时间，不把跨 batch 的 scheduler gap
任意归到某一桶。整体窗口吞吐仍用于比较相同工作负载、相同并发上限下的表现。

输出包括：

- `decode_throughput.csv`：三组 decode 对比、实际 batch 分布及 buffer 等配置。
- `throughput/<mode>/decode_steps.csv`：逐轮真实 batch、接受 token、GPU/CPU 时间和 gap。
- `throughput/<mode>/decode_by_batch_size.csv`：按实际 batch 分组的 GPU forward 指标。
- `throughput/<mode>/decode/rank-*.json`：各 rank 原始记录（schema 2，上限为 `max_batch_size`）。
- `decode/rank-*.status.json`：诊断状态，完成正式请求后的 flush 会强制刷新失败状态。
- `requests.json`、`cohort.json`：请求提交/首 token/结束时间及包含 prefill 的客户端耗时。
- `manifest.json`、`prompts.jsonl`、各组 `launch.json`、`server_info.json`、`server.log`。

客户端首 token 相对时刻由提交时刻加 TTFT 估算，仅辅助检查请求生命周期；实际
decode batch 及吞吐以服务端记录为准。`cohort.json` 记录 N、C 和补发策略；
`initial_wave_first_token_spread_s`、`initial_wave_token_stream_overlap_s` 只描述
最先提交的 `min(N,C)` 条请求。全部 N 条的 `first_token_spread_s` 包含后续补发
时间差，不能作为同一批请求的 TTFT 波动。

`benchmark_decode_batch_histogram` 统计预算筛选前的实际 batch；
`eligible_decode_forwards` 和 `longest_eligible_decode_streak` 统计合格 decode；
`rejected_forward_counts` 区分 prefill、输出余量不足、超出并发上限等原因。
`at_cap_decode_forwards_before_budget_filter=0` 或 `observed_max_decode_batch_size<C`
本身均不代表失败。旧版要求满 batch 的结果不要与新版混用，保留 manifest 中的 commit。

## 逐层 trace

需要定位 attention、MLP、refill/prefetch 等瓶颈时，另建输出目录，将运行命令加上：

```bash
--phase trace --trace-modes lagom hisparse dspark --nsys \
--output ./results/lagom-n16-c8-trace
```

三组使用相同 N、并发上限和 buffer 对照配置，trace 阶段也会完成 N 条正式请求。
Trace 只记录正式请求的实际 batch 在
1..C 范围内的 decode，跳过最初 2 个符合条件的 step 后记录 12 个；C=1 的服务预热
请求通过 request ID 排除。Prefill/idle 不采集。诊断 step 可以不连续，通过
`forward_iter` 检查间隔；逐层 trace/Nsight 耗时不能替代性能阶段。
`--phase all` 先跑三组性能，再用独立服务采集三组 trace。

32K 输入约有 8000 个 C4 KV，默认 hot=8192 几乎全驻留。测试中未出现预取收益时，
需结合真实换入量分析；缓存压力实验可另选更长输入或 `--w 3 --p 3 --hot 4096`，
并对两组 HiSparse 统一调整。

当前开发环境无 GPU，H100 上的容量、graph replay 和性能仍需实测。

```bash
python -m pytest -q test/registered/unit/observability/test_lagom_concurrent.py \
  test/registered/unit/observability/test_lagom_perf.py
```
