# V4 + DSPARK：扩展验证窗口的离线 probe 实验

本实验判断：**当前 DSPARK proposal 的后缀，在额外执行 target 层后，能否预测下一次实际验证窗口的历史 KV misses？**

分支：`exp/oasiskv-trace`，基于 `oasiskv` 的 `ef8bbeae7`。这里只增加 opt-in 采集入口，不启用在线多 token 预取，不改变原有 OasisKV 的 root-only 提交语义。

默认模型与硬件：

| 项目 | 配置 |
| --- | --- |
| Target | `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/` |
| DSPARK | `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| GPU | 4 × H100，TP=4，纯 TP，单机 |
| 数据 | `/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json` |
| 输入 | 16384 / 32768 token，沿用性能 harness 的 ShareGPT 拼接规则 |
| 首轮范围 | B=1，greedy，固定 V，无 grammar/logprob |
| 输出 / 采集窗口 | 每请求 256 输出 token；只记录前 32 个验证窗口 |
| KV | 原生 V4 + HiSparse；Top-K 读取 target 的 `index_topk`；驻留容量默认 `(G+1)×index_topk` |

**验证状态：本分支在无 GPU 环境进行了 CPU 单元测试；尚未在上述 checkpoint / 4 × H100 上实测。先跑下面的 smoke。** 需要使用已能运行该仓库 V4/DSPARK 的 CUDA、FlashMLA、DeepGEMM 等环境。脚本检查本地 checkpoint、GPU 和配置，不自动安装或下载模型依赖。

## 执行

在 GPU 机器的仓库根目录执行，使用原来运行 SGLang 的 Python 环境：

```bash
git fetch origin
git switch --track origin/exp/oasiskv-trace
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# 只读 checkpoint 配置并打印命令；不加载模型、不运行 GPU。
python benchmark/oasiskv/trace/run.py --output /srv/oasiskv/trace-plan

# 首先验证模型入口、快照恢复、接受边界和实际 miss 记录。
python benchmark/oasiskv/trace/run.py \
  --output /srv/oasiskv/trace-smoke \
  --input-lengths 4096 --output-length 32 \
  --requests 1 --max-windows 3 --skip-full-control --execute

# Smoke 成功后，再跑 16K / 32K。
python benchmark/oasiskv/trace/run.py \
  --output /srv/oasiskv/trace-v4-16k-32k --execute
```

已有本地实验分支时使用 `git switch exp/oasiskv-trace`。输出目录必须不存在，以防混合旧 trace。默认使用 GPU `0,1,2,3`；可通过 `--gpus` 修改。脚本顺序启动并关闭自己的服务进程，不连接或终止已有服务。每个模式共用同一份 `prompts.npy`，请求串行执行。

启动检查读取最终解析的 decode/prefill graph backend，两者必须都是 `disabled`，不依赖旧的 `disable_cuda_graph` 标志。
trace 在 `DSparkWorkerV2.alloc_memory_pool()` 完成 draft 池分配后构造，不在 worker 的 `__init__` 中构造。
实际启动顺序是：加载 target/draft 权重 → target 池分配及 HiSparse coordinator 创建 → draft 池分配 →
trace 校验/状态登记 → attention backend 初始化 → 服务接收请求。这里复用现有内存池生命周期，
不跳过 real-IO / 禁止在线预取检查，也不推迟到第一条请求才发现错误。
成功时 rank 0 日志包含 `OasisKV trace initialized after target/draft KV pool allocation.`，
对应模式目录出现 `runtime.json` 和初始空的 `trace.jsonl.gz`；此时尚不代表采集成功。
若 HiSparse 检查失败，异常中的 `failed` 列出具体失败条件，`runtime` 报告 coordinator/allocator 类型及各实际开关。
`enable_hisparse=True` 不等于已创建 V4 coordinator；`prefetcher=None` 也不等于旧 shared-index `enable_prefetch=False`。
该检查保持禁止在线预取及 skip-IO，不会为绕过启动错误而修改实验语义。检查成功后的状态也会写入 `runtime.json` 的 `hisparse_runtime`。

`eb91f01a4` 及之前版本在 worker 构造时过早读取尚未分配的 coordinator/allocator，会在这里失败。
升级后先用**新的输出目录**重跑上方 4K smoke；旧的 `manifest.json` 不是成功 trace，不可与新结果混用。
只有 control/probes 均产生有效窗口记录、`audit.json` 为 `passed=true`，且 `analysis/report.json` 有匹配样本，
才进入 16K/32K 正式采集。这个修复没有在本地 GPU 上验证，不据启动顺序回归宣称已有实验结果。

`G` 从 DSPARK checkpoint 的配置中读取，启动时明确传入 `speculative_num_draft_tokens=G+1`，**不扩大训练 proposal 窗口**。默认 `P=[1,2,3]`，`V=G−3`。例如 G=7 时 V=4。若 checkpoint 的 G 太短，脚本会报错；可选 `--verify-candidates 2 --probes 1 2`，但必须满足 `V+max(P)≤G`。控制组与 probe 组使用相同的 G 和 V。

Top-K 使用模型的原生设置，不用通用 Qwen 实验的 2048 代替 V4 的 indexer 配置。若 G=7、`index_topk=512`，默认驻留容量为 4096 个 C4 单元。现有普通 DSPARK 路径要求容量至少为 `(G+1)×index_topk`，以同时保留所有验证行的 KV；脚本遵守这一限制，不为制造 misses 而绕过它。三组容量相同，实际值写入 `runtime.json`。

可用参数见 `python benchmark/oasiskv/trace/run.py --help`。`--max-windows` 是每个请求的采集上限；不会伪造接受长度来凑窗口。若生成太短、没有连续两个验证窗口，分析会拒绝给出结果。想覆盖生成中后期，应增加 `--output-length` 和 `--max-windows`，并相应增加 `--context-length`。

## 三组数据

1. **control，P=0**：固定 V 的普通 DSPARK 接受流程，无 probe，无预测预取。
2. **probes**：同一个固定 V 基线轨迹，每个采集窗口执行 P=1/2/3 的 exact、resident 影子 forward；这些 forward 不进入接受流程、不提交 KV，也不执行后台预取。最后恢复并重放原来的 V 窗口，走普通 DSPARK 接受流程。
3. **full-control**：验证完整 G 个候选，无 probe。用于观察缩短 V 损失了多少接受长度；它与固定 V 的窗口划分不同，不强行逐窗口配对。Smoke 可以省略这一组。

每个 shadow 仍在**一次 target forward** 中计算 `[anchor, V 个验证候选, P 个 probe]`，让 probe 的 hidden state 经过所有 target 层。离线实验为做对照而多次重放；这不代表未来线上实现必须额外启动这些 forward。

P=0 没有新的预测集合。验证行自己的 Top-K 已经为本轮 attention 获取，不能把它重新统计成下一轮新增预取。

## exact / resident 的具体语义

- **exact**：所有 V+1+P 行使用原生稀疏 Top-K 和按需补载。它是相同 draft 分支的精确稀疏 attention 参考，不是 dense attention，也不是下一轮命中率的数学上界。
- **resident**：前 V+1 行照常补载。probe 先记录原始预测 Top-K，再仅映射其中已驻留的历史项，未驻留项的 attention 索引置为 −1；**不调用 probe 历史补载 kernel**。本窗口已计算的 C4 仍经过现有事务 scratch 映射和因果限制，SWA/C128 路径照常执行。

resident 的策略是“probe Top-K 与本层真实验证后驻留集合的交集”，不是把任意驻留项补满 K。这个选择会影响深层 hidden state，因此报告必须保留 mode 标签。记录 probe Top-K 的动作发生在这一层 attention 之前；下一层的 Top-K 已受前一层近似 attention 影响。

真正验证的接受规则沿用原来的 `accept_and_finalize`，用 compact layout 把接受边界限制为 V；最多提交 V+1 个 token。Probe logits 即使算出来也不参与基线接受判断。

## 隔离与正确性检查

离线采集使用**主机状态快照**：覆盖 target 的 SWA、C4/C128 KV、indexer KV、两类 compressor state、请求映射、HiSparse 驻留/LRU/scratch 表，以及 RNG 状态。原有 GPU 地址不变，恢复时原地写回。每次 shadow 前恢复同一个 pre-forward 状态；shadow 期间不执行 `commit_hidden` 或 C4 commit。

这不是线上预取的 staging buffer。没有增加线上读写双缓冲；快照只属于实验工具，代价不纳入性能结论。

会执行两层校验：

- 窗口内：shadow 前后的基线重放必须得到相同的 argmax token、Top-K、实际 miss 列表和驻留状态。
- 独立进程：`control` 与 `probes` 的完整输出 token ID、实际接受长度、验证输入、逐层 Top-K、miss 和驻留状态必须一致，否则停止分析。

另外检查所有仍被真实验证行引用的历史物理槽是否有效，避免逐行换入后，前面行的 KV 已被后面行淘汰。启动时已有容量下界检查，运行时若仍报 `references evicted C4`，**这次数据不能用于质量结论**，需要先检查状态映射或容量处理；不能忽略该检查继续分析。

每个 TP rank 单独保存一份 CPU 快照，`--snapshot-gib` 默认上限 32 GiB/rank，四卡最多约 128 GiB，另需为模型加载、host KV、框架保留主机内存。`snapshot-inventory.json` 给出实际大小。脚本限制 `max-total-tokens` 为 context length，避免按整张 GPU 的剩余容量分配巨大的实验 KV 池。内存不足时先检查这个文件；不要把 snapshot 上限当作实际预留量。

## 产物和指标

```text
trace-v4-16k-32k/
  manifest.json                 # 提交、硬件、checkpoint 配置指纹、参数
  source.diff                   # 启动时已跟踪文件的未提交改动
  prompts.npy / prompts.json    # 三组共享输入与数据/tokenizer 指纹
  control/, probes/, full-control/
    config.json / launch.json / runtime.json
    server.log
    trace.jsonl.gz              # 原始逻辑 C4 ID，只有 TP rank 0 写入
    outputs.jsonl              # 请求输出 token IDs
  probes/snapshot-inventory.json
  audit.json                    # 独立进程隔离检查，必须 passed=true
  acceptance.json               # 每种 V 的 A 分布、平均提交 token 数
  analysis/samples.csv           # 每窗口、每层、P、mode 的明细
  analysis/groups.csv            # 按输入长度、层、V、实际 A、P、mode 聚合
  analysis/report.json
```

trace 的 `layer` 是 C4 层序号；`runtime.json` 给出到原模型层号的映射。C4 单元代表 4 个原始 token，搬运字节取运行时 `item_size_bytes`，不把整个 C4 page 的 padding 算成每个选中单元的传输。

配对键为 `(run_id, request_id, generation, window_id, layer, head)`；window_id 是请求内的连续编号，**不按请求池槽号或全局 forward 计数配对**。要求下一窗口 `prefix_len = 当前 prefix_len + 实际 commit_len`。提前拒绝时也按这条真实路径配对，不能按 V 推测下一位置。

默认分析策略：probe Top-K 去重，仅保留源窗口已经提交的历史，再减去本层真实补载后的驻留集合；按被不同 probe 选中的频次排序，频次相同按逻辑 ID 排序，然后施加 `--prefetch-units` 上限。原生 Top-K 输出未保证按分数排序，因此不按数组位置做 rank 加权。

核心字段：

| 字段 | 含义 |
| --- | --- |
| `history_coverage` | 对下一窗口需要的、源窗口已有历史 KV 的覆盖率 |
| `prefetch_candidate_bytes` | 去重、扣除驻留、施加容量上限后的候选搬运量 |
| `next_miss_bytes` | 下一真实窗口 kernel miss plan 的实际 H2D 字节；重复补载按实际次数计 |
| `potential_unique_miss_coverage` | 候选集合覆盖下一窗口不同 miss 单元的比例 |
| `potential_first_fetch_bytes` | 潜在可提前获取的不同单元字节；同一单元多次补载只记一次 |
| `candidate_precision` | 候选中真正出现在下一窗口 miss 集合的比例 |
| `probe_demand_h2d_bytes` | exact probe 自己引入的补载；resident 必须为 0 |
| `next_unavailable_units` | 下一窗口使用、但源窗口尚未成为已提交历史的单元 |

没有用 kernel 调用次数代替搬运量；实际 miss 来自 swap kernel 输出的 `miss_src/miss_count`，再映射回逻辑 C4 ID。

16K 原始输入约对应 4096 C4 单元；若解析得到的容量是 4096，此组最初可能几乎没有 miss。若分母为零，指标输出 null，不伪造 100% 覆盖率，32K 组更有区分度。若 checkpoint 的 G 或 K 更大，安全容量可能覆盖这两档全部历史，报告会如实反映没有预取空间；需要另选更长输入研究容量压力。若调整容量，控制组和影子组必须一起重跑。

无需重跑模型即可扫描不同候选预算，例如：

```bash
python benchmark/oasiskv/trace/analyze.py \
  /srv/oasiskv/trace-v4-16k-32k/probes/trace.jsonl.gz \
  --max-prefetch-units 64 \
  --output /srv/oasiskv/trace-v4-16k-32k/analysis-cap64
```

这里的预算以 C4 单元计，是离线 admission 参数，不是通用路径 `max_new_blocks` 的论文推荐值。

## 这些结果不能说明什么

本轮是质量实验，**明确关闭 decode/prefill CUDA graph 和 overlap scheduler**，以便执行快照、可变宽度 shadow 和逐层观测。它不更改正常服务的 graph 配置。CPU 快照、GPU→CPU 观测和额外 forward 都会明显干扰时序，不能据此报告 TPS、TPOT、probe 的增量延迟或异步重叠收益；相应字段保持 null。

目前也没有把候选真正异步搬入下一窗口、模拟其间的淘汰、或观测传输完成时刻。因此 `potential_*` 只是预测潜力，`timely_miss_coverage` 和 `saved_critical_path_ms` 一律为 null。

下一步先看各层、各接受长度下 resident 对下一窗口**实际 misses**的覆盖，以及相对 exact 的变化；同时查看缩短 V 的接受长度损失。只有这些数据支持继续时，再实现线上融合 forward、KV 并集的安全生命周期和 graph 下的真实传输时序测试。

CPU 验证命令：

```bash
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 python -m pytest -q \
  test/registered/unit/spec/test_oasiskv_trace.py \
  test/registered/unit/spec/test_oasiskv_trace_startup.py
```
