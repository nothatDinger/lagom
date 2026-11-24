# Adaptive HiSparse 的 H2D 延迟实验

在 `perf/swapin_oneshot` 上，`run_overall.py --h2d-pair` 依次运行两个 DSPARK decode 实例。两次使用相同的输入、seed、CUDA Graph 档位和缓存容量，decode 的 `force_h2d_each_step` 分别为 `false` 和 `true`。默认是 P TP4 + D TP4 + router，32768 输入、1024 输出、C=256。H2D 配对配置指定 `W=5`，含 anchor 的最大验证宽度为 6。模型每个 query 选 512 行，hot 保留 2048 行，每个活跃请求另有 3072 行 staging。两组均设置 `resident_first=false`、`refill_pages_per_step=0`，使采样覆盖有 host 备份的 hot 命中。普通 PD 矩阵启用可选历史驻留；这两个参数也可在配对配置中一起调整。

在仓库根目录激活装有 SGLang 和 router 的 uv 虚拟环境。模型和 draft 的路径沿用 `overall.example.json`，可在 `--config` 指定的 JSON 中覆盖。自定义 H2D 配置请从 `h2d.example.json` 复制。先检查本地 ShareGPT 输入的命令生成，GPU 采集使用一台 8 卡节点。

```bash
source .venv/bin/activate
uv run --active --no-project python benchmark/lagom/run_overall.py --h2d-pair --dry-run \
  --sharegpt /data/ShareGPT.json --output /results/h2d

# 在本地 8 卡节点上运行两组测试
OUT="$PWD/results"
uv run --active --no-project python benchmark/lagom/run_overall.py --h2d-pair \
  --trace-steps 16 --graph-bs-cap 256 --sharegpt /data/ShareGPT.json --output "$OUT/h2d-latency"
```

预填充和解码各用 TP4。示例给 Mooncake 设置 `MC_FORCE_TCP=1` 和 `LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64`。本地 ShareGPT 按 seed 打乱对话并直接拼出定长 token IDs，语料不足会报错。也可传 `--data-dir /data/longbench-bank` 使用 `longbench_v2_fixed.py` 制作的 token bank。两种来源均支持 204800 输入。

默认在实际 decode batch 达到 256 后跳过两步，记录 12 步，排除启动预热请求。若机器容量使实际 batch 小于并发上限，`--trace-batch-size B` 可为两格指定相同的实际 batch。四个 TP rank 都要产出完整窗口，任一缺失会将该格标为失败。`--resume` 在相同协议下从最后成功的格继续。

`min_output_tokens=1023` 筛选的是请求的 `max_new_tokens`。正式请求输出 1024 个 token，prefill 可先产出 1 个，客户端预热请求最多生成 32 个。采样所需步数另算。W=5 时每步最多提交 6 个 token，跳过 2 步再采 12 步最多用 84 个，示例中的 16 步最多用 108 个。

## 如何核对解耦

每个 attempt 的 `steps/` 保留各 rank 的逐步时间戳、接受 token 数、前缀长度和 CSV。`h2d.json` 给出每个 rank 的每步均值，强制组的 `comparison.json` 给出两格差值。按 rank 比较，避免重复累计 TP 时间。

| GPU 区间 | 实际执行和计数 |
|---|---|
| `target.swap_in_gather` | 解析各 query 的有序选择，记录从 pinned host 复制到 arena 的唯一 C4 缺失行数 |
| `target.swap_in_reread` | resolve 将有 host 备份且符合重读条件的唯一 hot 命中计入 `resident_hits`。复制 kernel 在写完各行后累计 `kv_entries`。强制组写入 arena 空位，attention 继续使用原 hot 地址 |
| `target.swap_in_commit` | attention 读完后，LRU 将可接纳的缺失行从 arena 复制到固定 hot 地址。`admissions` 统计实际 D2D 行数，此段的 H2D `kv_entries` 为 0 |

每行 C4 payload 为 584 字节，三段的 `kv_entries` 之和乘 584 给出该 rank 每步的逻辑 H2D 字节数。验收读取 schema v2，逐层逐步核对基线实际复制 0 行、强制组实际复制数等于 eligible hits。`_mismatched_layers` 可检出不同层的误差相互抵消。旧 trace 的重读数由命中数推导，只能用于回看。源、目的地址和 KV 内容由对应的字节级 kernel 测试验证。

Adaptive 基线只保留两端的 GPU 打点。`h2d.json` 的 `reread_ms` 在该组为 0，`reread_span_ms` 单列空打点实测开销。整窗打点齐全时，`comparison.json` 中每个 rank 的 `delta.reread_ms` 为强制组区间减去基线空区间。基线缺少时间戳时，保留独立行计数，增量估计使用强制组原始区间并标记 `uncorrected_force_span`。该区间包括复制 kernel 的启动与 trace 计数开销，实际行数在复制循环结束后按线程块汇总。强制组及旧实现中确实执行扫描的基线都要求有效正时长。

`step_ms` 覆盖一个 DSPARK decode step。`decode_tpot_ms = 平均 step_ms / 每请求每步平均提交 token 数`。`reread_pct_of_step` 的分母是整个 step，`reread_ms_per_token` 已按提交 token 数摊销。客户端 `output_tps` 的计时包含长上下文预填充，逐步成本由上述 GPU 区间计算。

比较器逐 rank 检查新增 payload、接纳数、符合重读条件的命中数、原生 H2D 行数、commit 耗时和每步 token 数。后五项的组间相对变化容差为 10%。step 与重读增量须为正，两者差值的容差取基线 step 的 5% 和重读增量的 20% 中较大者。

`step_iqr_separated` 检查实测 step 分布的分离程度，要求强制组第 25 百分位超过基线第 75 百分位。单步窗口的这一项为 `null`，无法通过整体验收。一次 A/B 含两次独立启动，连续 step 可能相关，TP rank 也观测同一批请求。这项检查是描述性诊断。估计跨运行的统计显著性需要独立重复 A/B，并按运行对计算差值。

## Nsight Systems

在相同输入上另跑一组 `--nsys`，结果写入独立目录。在同一台 8 卡节点和 uv 环境中运行以下命令。

```bash
OUT="$PWD/results"
uv run --active --no-project python benchmark/lagom/run_overall.py --h2d-pair --nsys \
  --trace-steps 16 --graph-bs-cap 256 --sharegpt /data/ShareGPT.json --output "$OUT/h2d-nsys"
```

只有 decode 服务在 `nsys profile` 下启动。PD 使用 P TP4 + D TP4，报告应覆盖 4 个 decode rank。第一个采样 step 由 decode rank 0 调用 `cudaProfilerStart()`，各 rank 再进入 step。最后一个 step 的主流和已注册侧流完成后，各 rank 汇合，由 rank 0 调用 `cudaProfilerStop()`。启动参数使用 `--capture-range=cudaProfilerApi --capture-range-end=stop`，服务继续处理剩余请求。采集范围为 CUDA/NVTX。NVTX 标记用于关联 step，`--cuda-graph-trace=node` 展开图内 kernel，`--sample=none --cpuctxsw=none` 允许在受限主机采集。按名称触发 torch NVTX 在 nsys 2025.1.3 上曾静默产生空报告，使用 `--delay` 和 `--duration` 则可能采到模型加载期。`stop-shutdown` 会终止服务。

默认 decode CUDA Graph 的最大 batch 是该格的并发上限。`--graph-bs-cap` 可以进一步限制档位，实际采样 batch 必须能被图覆盖。C=256 时最大图为 256。profiled decode 使用 `--watchdog-timeout 100000` 秒、`--dist-timeout 3600` 秒、`TORCH_NCCL_TIMEOUT=3600000` 毫秒和 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` 秒。当前分支将 `dist_timeout` 传入默认进程组和模型并行 NCCL 子组。

每格保留 `prof.nsys-rep`。runner 等待 profiler 收尾，在可写临时目录复制报告并执行 `nsys stats --force-export=true`，验证有正时长的 CUPTI kernel 后保存 `prof.sqlite`。缺失或空的 rep、缺少任一 decode rank 或采样步、缺少 fused kernel 均会使该格失败。`nsys-stats.txt` 是 nsys 的原始跨设备汇总。

| 产物 | 读数 |
|---|---|
| `nsys-steps.json` | 每 rank、每 step 的 GPU 时间线。kernel、显式 H2D、其他 GPU 活动、跨类别重叠和 GPU 空闲的互斥时长之和等于采集到的 step 跨度 |
| `nsys-kernels.json` | 每 rank 在所选 steps 内的 demangled 名称、实例数和 kernel 耗时。类别百分比的分母为该 rank 的 kernel 实例时长总和，允许并行 kernel 重叠 |
| `nsys-comparison.json` | 强制组目录中两格相同 rank、设备、步数的对比。每类 kernel 报告整个窗口的基线、强制和差值，单位为 ms |
| `nsys-workload.json` | 每 rank 的原生 H2D 行数、额外重读行数、候选和接纳数。按实际进程 PID 将 trace 与 CUPTI 设备对应 |
| `h2d.json`、`comparison.json` | 单独的 Lagom trace 延迟运行产出，提供 step、decode TPOT、重读增量和配对验收结果 |

开启 nsys 时，`summary.csv` 保留请求完成数和作业状态，吞吐及 TPOT 栏留空。原始客户端输出保存在 `benchmark.jsonl`。

CPU CUDA API 按该 step 的 scheduler 调用线程取区间并集。GPU 空闲还细分出与这些 API 重叠的时间和其余空闲时间，可用于观察 launch 间隙。完整 API 并集可能与 GPU 工作并行。显式 H2D 字节数只累计完整落入 step 的 memcpy，并列出跨边界次数。fused kernel 对 pinned host KV 的读取属于 kernel 活动，其逻辑 payload 由 Lagom 行计数乘以 584 B 得到。

`dhisparse_plan_kernel` 和 `DHiSparseCopy::Refill` 归入 `swap_in_gather`，`DHiSparseCopy::Reread` 归入 `swap_in_reread`，`dhisparse_admit_kernel` 和 `DHiSparseCopy::Promote` 归入 `swap_in_commit`。基线没有额外的 reread kernel。旧分支的 `sglang::staging::promote_kernel`、`copy_kernel<true>`、`copy_kernel<false>` 各有独立分类。GEMM 与其余 kernel 保留原名供复核。

对已有报告，可将输出指向可写目录。`--steps` 使用本分支的 step 标记，`--kernels` 支持仅有 CUDA kernel 的历史报告，并以整个采集窗口为范围。

```bash
uv run --active --no-project python benchmark/lagom/analyze_nsys.py /archive/prof.nsys-rep \
  --steps --output /tmp/h2d-steps.json
uv run --active --no-project python benchmark/lagom/analyze_nsys.py /archive/old.sqlite \
  --kernels --output /tmp/h2d-kernels.json
```

在 8×H100 的历史采集中，每步耗时从约 120 ms/step 增至 6.6 s/step，约 55 倍。采集窗口中的 GPU 利用率约 37%。nsys 提供采集条件下的 kernel 相对组成，CUPTI 背压会改变调度和重叠。旧实现的 promote/refill/stage copy 约为每 rank 每 16 步 72/5.0/3.9 ms。本分支的实际增量需用相同数据和 batch 重新测量。绝对 step 和 decode TPOT 取自独立的低开销 trace，端到端吞吐取自关闭诊断的 serving 运行。

## perf/lagom 原有实验

8×H100、Mooncake P TP4 / D TP4、LongBench-v2 的 32k/64k/128k × 并发 8–256 overall TPS/TPOT 实验，见 [overall 实验说明](OVERALL.md)，入口为 `run_overall.py`。该方案区分无推测解码的 Hisparse、DHisparse 和 Lagom；Lagom PD 状态布局兼容及执行环境验证事项也列在该文档中。

在 `perf/lagom` 分支运行，目标是快速判断预取是否带来收益，以及时间花在哪里。
这是短测方案，不是稳定吞吐结论或生成质量评测。需要能运行 DeepSeek V4 和配套
DSPARK 的 CUDA GPU、模型依赖及本地 ShareGPT JSON。当前开发环境没有 GPU，
这些实验尚未实际执行。

并发上限 8/16/32/64、按实际 running batch 统计 decode 性能的测试，见
[高并发实验说明](CONCURRENCY.md)，入口为 `run_concurrent.py`，可用 `--requests`
独立设置正式请求总数。

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
- 默认 `W=7, P=4, N=21, K=512, hot=8192`。验证 arena 容量为 6144 行，
  与 hot cache 独立；两个 HiSparse 组均使用相同的 8192 个 hot C4 槽位。
- 32k 输入约有 8000 个 C4 KV，在这一配置下几乎全驻留，实验未出现加速不能
  直接归因于预取无效。需要补充缓存压力实验时，可显式使用 `--w 3 --p 3 --hot 4096`。
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

先确认 target 和 draft 的 **routed expert 权重格式**。DeepSeek V4 的
`quantization_config.quant_method=fp8` 可以与 `expert_dtype=fp4` 同时存在；
应结合权重头与启动日志中的 `is_fp4_experts` 判断，不能仅凭 quant_method。
普通 TP 下 `auto` 可能选择 FP8 Triton runner，它不支持 packed MXFP4。

H100/H200（SM90）上，若 target 和 draft 均使用原生 MXFP4 专家，两者显式设置
`marlin`。两种后端参数会用于全部吞吐组和 trace 组；不指定时仍保留服务本身的
默认选择，脚本不会自动转换权重格式。若 draft 实际为 FP8，给 draft 单独使用
`--speculative-moe-runner-backend auto`。其他 GPU 需选择相应的 MXFP4 后端，
例如 SM100 的 `flashinfer_mxfp4`，不能直接套用 Hopper 的 Marlin 配置。

对于本次 H100、TP=4 的 FP4 target/draft，在已安装本仓库及 CUDA 依赖的环境执行：

```bash
python benchmark/lagom/run.py \
  --model /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --draft /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --sharegpt /data/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tp 4 --w 7 --p 4 --hot 8192 \
  --moe-runner-backend marlin --speculative-moe-runner-backend marlin \
  --output /results/lagom-quick-01
```

该 checkpoint 的 `dspark_block_size=5`，这里按实验要求显式覆盖为 W=7；
接受长度应随结果记录。之前启动命令中的 `num_draft_tokens=6`、block size=5、
prediction_steps=5 对应 W=P=5，与此次 W=7、P=4 的配置不同。

脚本顺序启动三个未插桩的服务做吞吐测试，再单独启动一个带插桩的 Lagom 服务
采集 trace，退出后清理自己启动的进程。默认共 4 次模型加载，不同时占用两份模型。
输出目录必须不存在，避免覆盖之前的实验。保持 GPU 空闲，不混跑其他负载。

只测吞吐，减少一次模型加载：

```bash
python benchmark/lagom/run.py --model /models/DeepSeek-V4 \
  --draft /models/compatible-DSPARK --sharegpt /data/sharegpt.json \
  --tp 4 --moe-runner-backend marlin --speculative-moe-runner-backend marlin \
  --phase throughput --output /results/lagom-throughput-01
```

为三组都采集 trace，并启用 Nsight Systems 的关联 CPU/CUDA 时间线：

```bash
python benchmark/lagom/run.py --model /models/DeepSeek-V4 \
  --draft /models/compatible-DSPARK --sharegpt /data/sharegpt.json \
  --tp 4 --moe-runner-backend marlin --speculative-moe-runner-backend marlin \
  --phase trace --trace-modes lagom hisparse dspark --nsys \
  --output /results/lagom-trace-01
```

需要安装支持 `--cuda-graph-trace=node` 的 `nsys`。采集由 cudaProfilerStart/Stop
限制在采样窗口附近，避免把整个模型加载写进 report；rank 0 控制采集范围。
所有 rank 的轻量 GPU trace 都会独立导出。默认未开启 Nsight。

其他参数：`--w`、`--p`、`--prefetch-slots`、`--hot`、`--top-k`、`--trace-steps`、`--trace-skip`、
`--mem-fraction`、`--chunked-prefill`、各 timeout。`--dry-run` 只显示启动命令，
不读取模型、不启动服务。额外相同的模型参数可通过 `--server-args-json file.json`
提供字符串数组；不能覆盖脚本控制的实验开关。模型 K 会从配置读取并与 `--top-k`
核对；容量不足直接报错，不静默修改缓存大小。

如果旧版本在 graph capture 时出现 `fused_moe.py: Hidden size mismatch`，并且
日志同时显示 `is_fp4_experts=True` 与 Triton runner，原因是 FP4 专家权重以
`hidden_size/2` 打包存储，而 FP8 Triton 要求完整 K 维。这与 W/P 或 trace 无关。
本分支会在初始化该 runner 时提前拒绝此组合；正确修复是选择兼容的专家后端。
保持 CUDA Graph 开启，且不要通过 `SGLANG_DSV4_FP4_EXPERTS=0` 将原生 FP4
权重误标为 FP8。日志中的 FlashInfer deferred finalize disabled 本身不是错误。

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

`prefix_lens` 记录每轮 forward 开始前的实际序列长度。纯 DSPARK 的
`seq_lens_cpu` 可以为空，此时在 GPU 上保留长度快照，采样结束且完成事件就绪后
才读取到 CPU 并导出；不会在采样轮内为此增加 D2H 同步。

| 字段/区间 | 含义 |
|---|---|
| `step_ms` | draft、verify、accept/commit/inject 的主流总跨度，包含其间的等待和 GPU 空洞 |
| `draft_ms` | draft proposal 全过程，包括模型执行和候选处理 |
| `draft.model_ms` | draft 模型 forward 本身（不含外部候选采样） |
| `target_verify_ms` | target verify 全过程 |
| `target.kv_plan_ms` | 两个 HiSparse 组的整窗口 staging 去重与寻址 |
| `target.kv_refill_ms` | 两个 HiSparse 组剩余真实缺页的主流 H2D |
| `target.kv_stage_copy_ms` | hot / prefetch / writer 命中汇入共享验证 arena 的 D2D |
| `target.kv_promote_ms` | attention 后已提交历史行的 hot-cache 晋升 D2D |
| `target.kv_refill_fused_ms` | 旧 swap-in fallback 的融合区间；当前 DSPARK staging 路径不使用 |
| `target.kv_prefetch_ms` | 本轮产生的侧流预取工时，按层求并集 |
| `target.prefetch_wait_ms` | 当前主流等待上一轮同层预取完成的暴露时间，属于消费轮 |
| `target.attention_core_ms` | V4 attention backend 执行，含相关 KV store 与 attention kernel |
| `target.attention_block_ms` | attention 模块完整跨度，包含 Q/K/V 投影、indexer、refill 等；与上面的字段嵌套 |
| `target.mlp_ms` | MLP/MoE，包括 routing 和通信 |
| `cpu_dependency_commit_ms` | accepted length 取回 CPU 及依赖它的 CPU 处理区间；含 D2H 和等待，不能直接解释为 GPU 等 CPU |
| `gpu_wait_cpu_ms` | 默认未知/null；没有 CUPTI 证据时不把剩余时间当作 CPU 阻塞 |
| `inter_step_gap_ms` | 上一轮主流结束到本轮开始的间隔；包含 scheduler 等未归因开销，不直接等同 CPU 阻塞 |

GPU region 的 `kv_entries` 记录当前 rank 实际搬运的 C4 行数，分为主流 H2D
`target.kv_refill`、主流 D2D `target.kv_stage_copy` 和侧流 H2D
`target.kv_prefetch`。计数基于去重后的搬运计划，排除无效条目和 graph padding。
侧流计数归到发起预取的 step。每行包含 576 字节的 KV 数据和 8 字节的量化
scale，故 payload 字节数为 `kv_entries * 584`。JSON 的 summary 和
`.steps.csv` 按 step 对各层计数求和，分别输出 `target.kv_refill_entries`、
`target.kv_stage_copy_entries` 和 `target.kv_prefetch_entries`。Chrome trace
在对应 GPU 区间的 args 中显示 `kv_entries`。执行了搬运阶段但没有数据可搬时，
计数为 0；没有对应采样区间时，CSV 留空。

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


当前先测试 N=21 的整步预取。prefetch 在本层 attention 和晋升结束、槽位释放后发起；
下一轮到该层时等待完成。W=7/P=4/K=512 时预取 payload 约 24 MiB/请求/GPU。
--prefetch-slots 可配置，但本阶段必须覆盖全部 CSA 层；N<层数的滚动调度尚未实现。
各组保留相同 hot 配置，并把 staging D2D / 晋升计入总成本；新路径尚需 GPU 验证。
