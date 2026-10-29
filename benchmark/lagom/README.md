# Lagom 初步性能实验

在 `perf/lagom` 分支运行，目标是快速判断预取是否带来收益，以及时间花在哪里。
这是短测方案，不是稳定吞吐结论或生成质量评测。需要能运行 DeepSeek V4 和配套
DSPARK 的 CUDA GPU、模型依赖及本地 ShareGPT JSON。当前开发环境没有 GPU，
这些实验尚未实际执行。

## 三组对比

| 模式 | DSPARK | HiSparse | Lagom 预取 |
|---|---|---|---|
| `lagom` | W 个解码候选，加 P 个预测候选 | 开 | 开 |
| `hisparse` | W 个解码候选 | 开 | 关 |
| `dspark` | W 个解码候选 | 关 | 关 |

`dspark` 仍使用 V4 原生 CSA，只关闭 HiSparse 分层 KV 管理。所有组固定 PD 融合、
concurrency=1、TP、采样参数、W、prompt 和 CUDA Graph；两个 HiSparse 组固定
相同 hot 容量，均为 `force_h2d_each_step=false`。开启 `ignore_eos`，保证实际
输出长度；关闭 radix cache，避免不同启动顺序造成前缀命中差异。

默认实验：

- 输入 **32000 token**，输出 **1024 token**；每组 1 个 32-token 输出预热请求，
  然后测量 **2 个完整请求**。可以用 `--requests 1` 做更短 smoke test。
- 默认 `W=3, P=3, K=512, hot=4096`。并集上界 `(W+P+1)*K=3584`，可以放入
  hot 缓存；32k 输入约有 8000 个 C4 KV，因此能形成缓存压力。
- 如果设 `W=7,P=7`，必须用 `hot>=7680`，例如 `--hot 8192`。这时 32k 输入
  几乎全驻留，实验未出现加速不能直接归因于预取无效。
- Decode Graph 只 capture batch size 1，减少启动时间，使用 static verify。
- 默认额外跑 **Lagom 一组 trace**：输入仍为 32000，输出默认 160；跳过前 2
  个 verify step，再记录 12 个 step。W 较大时自动增加 trace 输出长度，确保
  足够步数以及异步导出。trace 请求不参与吞吐统计。
- 每次启动后默认运行预算 600 秒，单请求 180 秒；模型启动预算 900 秒。
  checkpoint 加载、JIT 和 graph capture 不计入吞吐，也不能保证在短时间内完成。

标准 ShareGPT 并不保证有 32k 长输入。脚本固定 seed 打乱并拼接真实对话，按目标
模型 tokenizer 截成恰好 32000 个 input_ids，不使用随机 token、文本填充或 chat
模板。结果应称为 **packed ShareGPT，固定长度**，不是原始 ShareGPT 长度分布。
三个模式使用完全相同的 token IDs，并保存源记录编号和 token hash。

## 运行

先进入仓库根目录，在已安装本仓库及 CUDA 依赖的环境执行：

```bash
python benchmark/lagom/run.py \
  --model /models/DeepSeek-V4 \
  --draft /models/compatible-DSPARK \
  --sharegpt /data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tp 8 --output /results/lagom-quick-01
```

脚本顺序启动三个未插桩的服务做吞吐测试，再单独启动一个带插桩的 Lagom 服务
采集 trace，退出后清理自己启动的进程。默认共 4 次模型加载，不同时占用两份模型。
输出目录必须不存在，避免覆盖之前的实验。保持 GPU 空闲，不混跑其他负载。

只测吞吐，减少一次模型加载：

```bash
python benchmark/lagom/run.py --model /models/DeepSeek-V4 \
  --draft /models/compatible-DSPARK --sharegpt /data/sharegpt.json \
  --tp 8 --phase throughput --output /results/lagom-throughput-01
```

为三组都采集 trace，并启用 Nsight Systems 的关联 CPU/CUDA 时间线：

```bash
python benchmark/lagom/run.py --model /models/DeepSeek-V4 \
  --draft /models/compatible-DSPARK --sharegpt /data/sharegpt.json \
  --tp 8 --phase trace --trace-modes lagom hisparse dspark --nsys \
  --output /results/lagom-trace-01
```

需要安装支持 `--cuda-graph-trace=node` 的 `nsys`。采集由 cudaProfilerStart/Stop
限制在采样窗口附近，避免把整个模型加载写进 report；rank 0 控制采集范围。
所有 rank 的轻量 GPU trace 都会独立导出。默认未开启 Nsight。

其他参数：`--w`、`--p`、`--hot`、`--top-k`、`--trace-steps`、`--trace-skip`、
`--mem-fraction`、`--chunked-prefill`、各 timeout。`--dry-run` 只显示启动命令，
不读取模型、不启动服务。额外相同的模型参数可通过 `--server-args-json file.json`
提供字符串数组；不能覆盖脚本控制的实验开关。模型 K 会从配置读取并与 `--top-k`
核对；容量不足直接报错，不静默修改缓存大小。

## 结果

- `manifest.json`：参数、commit、输入格式。
- `prompts.jsonl`：固定 input_ids、源记录与 token hash；包含数据集内容，按本地实验数据管理。
- `throughput.csv`：三组输出吞吐、decode 吞吐、TTFT、TPOT。
- `throughput/<mode>/requests.json`：每个请求的耗时、实际长度、接受长度及输出 hash。
- 每组 `launch.json`、`server_info.json`、`server.log`：实际启动参数与服务信息。
- `trace/<mode>/steps/rank-*.json`：逐 step、逐层 GPU/CPU 时间及接受长度。
- `*.steps.csv`：每个 verify step 的耗时摘要。
- `*.trace.json`：可在 Perfetto UI 或 Chrome tracing 中打开的时间线。
- `trace/<mode>/nsys.nsys-rep`：可选的 CUPTI/CUDA API/NVTX/CPU 关联 trace。

`output_tokens_per_s` 是总输出 token / 测量请求总延迟，包含 prefill；
`decode_tokens_per_s` 只计算第一个流式响应之后的输出 token / 后续时间。
第一个响应可能包含多个推测接受 token，因此它是流式近似 decode 指标。
不把预热、服务启动、trace 请求计入吞吐。2 个请求不足以做可靠 p99 统计；初测
比较吞吐、逐请求 TTFT/TPOT 和接受长度，发现收益后再增至 10+ 请求并重复实验。

## 每个 verify step 如何分解

GPU 标记在 graph 内使用 `%globaltimer` 写固定的 `[region,step,edge]` 缓冲区，
每次 replay 使用新的 step 下标，侧流保存该层自己的生产 step 下标。采样过程中
不为读取时间戳增加逐 step synchronize；结束后等待事件 query 完成才导出。
标记 kernel 会扰动运行，所以 **详细 trace 不能用于报告无插桩吞吐**。

| 字段/区间 | 含义 |
|---|---|
| `step_ms` | draft、verify、accept/commit/inject 的主流总跨度，包含其间的等待和 GPU 空洞 |
| `draft_ms` | draft proposal 全过程，包括模型执行和候选处理 |
| `draft.model_ms` | draft 模型 forward 本身（不含外部候选采样） |
| `target_verify_ms` | target verify 全过程 |
| `target.kv_plan_ms` | Lagom 的并集槽位规划 |
| `target.kv_refill_ms` | Lagom 当前窗口的主流缺页搬运 |
| `target.kv_refill_fused_ms` | 非 Lagom HiSparse 的融合 gather/commit，包含规划与搬运，不能称为纯 copy 时间 |
| `target.kv_prefetch_ms` | 本轮产生的侧流预取工时，按层求并集 |
| `target.prefetch_wait_ms` | 当前主流等待上一轮同层预取完成的暴露时间，属于消费轮 |
| `target.attention_core_ms` | V4 attention backend 执行，含相关 KV store 与 attention kernel |
| `target.attention_block_ms` | attention 模块完整跨度，包含 Q/K/V 投影、indexer、refill 等；与上面的字段嵌套 |
| `target.mlp_ms` | MLP/MoE，包括 routing 和通信 |
| `cpu_dependency_commit_ms` | accepted length 取回 CPU 及依赖它的 CPU 处理区间；含 D2H 和等待，不能直接解释为 GPU 等 CPU |
| `gpu_wait_cpu_ms` | 默认未知/null；没有 CUPTI 证据时不把剩余时间当作 CPU 阻塞 |
| `inter_step_gap_ms` | 上一轮主流结束到本轮开始的间隔；包含 scheduler 等未归因开销，不直接等同 CPU 阻塞 |

区间是 inclusive span，可能包含嵌套和跨流重叠，**不能直接相加当成总延迟**。
Graph replay 的内部 Python 不会重新执行，内部 GPU 标记仍会执行。CPU 轨迹与
GPU 轨迹在轻量 Chrome 导出中采用独立零点；不能用这两个时钟直接判断因果。
用 Nsight 的关联时间线确认 CPU 提交空洞、D2H 同步、CUDA event 等待和 NCCL。

预取总工时不等于暴露时间：prefetch 的生产轮为 i，等待它的消费轮为 i+1。
跨轮看同一层 `kv_prefetch` 与 `prefetch_wait`，检查是否隐藏在当前 verify 的
后续层和下一次 draft 中。第一轮可能没有上一轮预取，不能仅看第一轮。

## CPU 数据依赖等待的保守统计

Nsight 可视化时展开 `lagom/step/step=N` 与 `lagom/cpu_dependency_commit/step=N`
的 NVTX 区间，检查 GPU producer、D2H、CPU 处理和下一次 launch。额外导出：

```bash
nsys export --type sqlite --output /results/lagom.sqlite \
  /results/lagom-trace-01/trace/lagom/nsys.nsys-rep
python benchmark/lagom/analyze_nsys.py /results/lagom.sqlite \
  --output /results/lagom-cpu-dependency.json
```

脚本输出显式 CPU 依赖区间内 **所有被采集 GPU 都没有 kernel/memcpy/memset**
的时间。它使用 Nsight 的同一时钟，是 TP 下的保守下界，不是全部 CPU 引起的
GPU 等待；不要跨 TP rank 求和。纯 DSPARK 没有 HiSparse commit 区间时结果可以
为空，应从完整 NVTX/CUDA API 轨迹进一步定位其他 CPU 依赖。

## 初步判断顺序

1. 检查实际输入/输出长度、每组配置及 `server.log`，确认没有禁用 CUDA Graph；
   trace 会检查每个采样 step 的 draft 和 target graph 标记及关键 GPU 区间，
   发现 eager fallback 或缺失标记即报错。
2. 对比三组吞吐和接受长度；输入、W、hot、温度不同的结果不能直接比较。
3. Lagom 若慢，先看额外 P 的 draft/target 计算成本，再看 refill 是否减少、
   预取是否仍有较多暴露等待，以及 MLP/通信是否已占主要时间。
4. 用 Nsight 核对 H2D 与 compute 是否重叠、是否带来带宽竞争，再判断 CPU 提交
   空洞。不能只凭平均 KV copy 时间就声称端到端加速。

开发环境可运行不需要 GPU 的脚本/统计测试：

```bash
python -m pytest -q test/registered/unit/observability/test_lagom_perf.py
```
