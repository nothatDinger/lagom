# OasisKV 性能测试准备：Fig.11

本目录基于 `oasiskv` 的提交 `333551cf8f6884d1a9eef30dfd70932cd547d9c4`，
为 [OasisKV v1](https://arxiv.org/pdf/2608.08097v1) §5.2、§5.3.1 / Fig.11
准备长度、并发扫描与指标采集。**尚无 GPU 性能结果。**

## 测试矩阵

| 项目 | 配置 |
| --- | --- |
| Target | `/mnt/public_data/Qwen/Qwen3-8B`；`/mnt/public_data/Qwen/Qwen3-235B-A22B` |
| GPU / TP | 8B：1 × H100 80GB / TP=1；235B：8 × H100 80GB / TP=8 |
| Draft 本地目录 | 8B：`/mnt/public_data/Qwen3-8B_eagle3`；235B：`/mnt/public_data/Qwen3-235B-A22B-Eagle3` |
| 输入 / 输出 | 16,384、32,768 / 2,048 token；上下文上限 40,960，留出 lookahead 空间 |
| 最大并发 | 8、16、32、64、128、256；同时设置客户端并发与服务端 `max-running-requests` |
| 方法 | SGLang dense FA3；本分支 OasisKV |
| 稀疏配置 | 16-token block，128 个历史块（2,048 token），额外 active block；`max_new_blocks=null` |
| 采样 | greedy，固定 seed=42，`ignore_eos=true`，以服务端输出 token 计数验证长度 |
| 数据 | `/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json` |
| 本地测试协议 | 每点 1,024 请求、3 次重复；每次独立启动服务、2 个 32-token warmup 请求 |

全矩阵有 48 种配置 × 3 次重复 = 144 个测试点。正式测量总输出约 3.02 亿 token，
235B 每次启动还要重新加载权重；应先执行 smoke。1,024 请求保证最高并发至少有四倍请求量，
它和重复次数、warmup、`mem_fraction_static=0.9`、`chunked_prefill_size=8192` 都是本测试明确选定的值，
不是论文披露的默认值。每个点记录实际启动参数与服务端解析后的配置。

论文主机为双 Xeon 8480C、2 TB RAM，GPU 间 NVLink/NVSwitch，主机到 GPU 为 PCIe Gen5 ×16。
preflight 检查可见 H100 数量和显存，并保存 CPU、内存和 GPU 拓扑供核对。
`config.json` 的 CUDA 设备编号可按目标机器调整；两模型按顺序执行，不同时争用 GPU。

## 与论文的边界

当前 decode CUDA graph 状态（配置与代码核对，尚未在 GPU 上验证 capture/replay）：

| 测试方法 | Decode CUDA graph |
| --- | --- |
| SGLang dense / FA3 | 显式配置 decode `full`，`max_bs` 为该点并发上限。实际 capture 与 replay 需看运行日志。 |
| 通用 OasisKV（本次两个 Qwen 模型） | 显式配置 decode `full`，`max_bs` 为该点并发上限；target paired decode 和 EAGLE-3 draft 不再被强制关闭 graph。新增通用路径尚待 GPU capture/replay 实测。 |

通用 decode 通过固定 GPU 请求表间接访问各请求的 host KV、摘要和 device working set；
请求身份、长度和扩容后的地址在 replay 前更新。图内执行 root KV 追加、摘要更新、预测、
真实 Top-K 补载和成对 attention。每层的选择/传输使用独立捕获的 CUDA graph，
由 external event 连接：选择等预测及真实补载，传输再等本层 attention；前台图尾不汇合后台。
提交前台图后立即提交各层后台图，下一步到达对应层才等待该层上次传输，允许末层传输跨过模型边界。
后台保存独立元数据快照，避免下一批请求换序或 padding 更新覆盖正在执行的任务。
host 分配/扩容、dense→sparse 转换和请求回收仍在图外，必要时等待相关页的使用者。
变长 prefill 保持 eager；短请求直接使用原有 paged extend attention 读取共享页池。
`--disable-cuda-graph` 保留为显式 eager 对照；正式测试须确认日志中的实际 capture/replay。
性能脚本检查服务端解析后的 decode graph 配置；若关闭 graph 或最大 batch 不覆盖该点并发，会将该点判为失败。

先在目标 CUDA 环境从仓库根目录执行无需 checkpoint 的图回归测试，再进行服务 smoke：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/cuda_graph/test_oasiskv_async_graph.py \
  test/registered/cuda_graph/test_oasiskv_cuda_graph.py \
  test/registered/cuda_graph/test_oasiskv_v4_cuda_graph.py
```

测试覆盖连续 replay、不同请求顺序、padding、host 扩容、请求槽复用，以及 MHA/GQA、MLA 和原生 DSA 的 KV 路径。
它验证 attention/cache 内核及多流图依赖；原生 DSA 的实际模型 indexer 和 EAGLE-3 worker 仍需完整服务 smoke 验证。

- 原论文使用 vLLM V1 v0.12.0；dense 使用 FA3/full CUDA graphs，OasisKV 使用 piecewise graphs。
  此处是 **SGLang 移植版与同框架 dense 的对照**。当前通用 OasisKV 使用 full decode graphs，预测在主流、
  选择/传输在后台；论文的 C++ worker 和三条后台流水并不等同于当前实现。
- 按本分支的既有要求，每步真实 query 都补载 miss，draft 只预测与预载、强制拒绝。
  `max_new_blocks` 仅约束后台预取，不限制真实 query 补载；不能将其等同于论文 capped resident attention。
  Fig.11 没有清楚指定 fetch cap，§5.5 的 0.05 是消融设置，因此这里不把 0.05 宣称为 Fig.11 默认值。
- 论文性能部分描述合成长度负载，没有指明 ShareGPT。这里按用户要求使用 ShareGPT 内容：
  将所有有效轮次按固定 seed 打乱，串接 token 流并切成长度为 32,768 的行；16k 取同一行的前缀。
  不套 chat template、不添加 BOS，直接发送 `input_ids`，避免 decode/re-tokenize 改变长度。
  这用于固定长度性能测试，不代表自然长对话分布，也不用于质量评估。
- 本目录未实现 ShadowKV、InfiniGen、FreeKV 或原生 vLLM 的自动对照。
  后续加入它们时需要各自独立环境、同一 token 数据与相同指标定义，不应把 dense 曲线标成 vLLM。
- GPU 上的正确性、启动兼容性、显存与 pinned host 内存容量均须通过 smoke 验证；CPU 测试通过不代表可达到论文速度。
  高并发容量不足或 OOM 应保留为失败点，不自动调低并发、长度或预算。
  当前通用路径按需从共享物理页池分配 KV 页，但容量估算仍按 `max-running-requests` 预留最坏 working set，因此高档并发也可能在启动阶段容量不足；
  不能假定它一定像论文曲线那样自动维持较低的 running batch。服务端报告的实际请求上限和 KV 容量会留存。


### 本轮补齐的机制

- **跨步 graph 流水**：通用与 V4 路径都移除了模型尾部的侧流汇合。V4 的 C4 提交只等元数据选择读完，短序列后台不回写旧映射，历史传输不写 reserved newest slot。
- **成对 KV 复用**：通用 CUDA decode 一个 CTA 同时处理真实行和 draft 行，每个历史 K/V tile 只加载一次，分别累计 softmax；真实行仍屏蔽 draft KV。eager 和 full graph 使用同一个内核。
- **统一分页及阈值切换**：通用模型的 dense/sparse 请求共享每层物理页池和 head-wise 页表。长度不超过 `top_k` 时按需分配 dense 页，无 host history；超过阈值时将完整历史写入 pinned host，已有 GPU 页原地转为 sparse 槽位。不同请求可处于不同模式，同一 decode graph 处理两种行。页回收后可供其它请求复用。

这套页池服务于 OasisKV target；它没有启用 SGLang 的独立 `--enable-unified-memory` 选项，也不与 draft 权重或 KV 共用物理页。长请求的 sparse prefill 仍临时恢复完整 KV 进行 causal attention，调度容量估算仍保留最坏情况预算。
本轮已做 CPU/Triton 数值回归和 SM90 离线编译；新增 external-event 跨步依赖尚待真实 CUDA capture/replay 验证。不能据此宣称 TPOT/TPS 已改善。

### 补传与预传的去重判据

同一步、同一请求代次、同一层、同一 KV head 内，预取应计算：

```text
预取 miss = draft 预测块集合 − 本步真实 query 补载后的完整驻留集合
```

通用 full graph 在补载后的 `publish_plan` 内，把完整驻留 ID/age 写入本层后台计划的既有缓冲区；后台等 `prediction_ready` 后，用这份基线计算 miss。基线不引用后续层会复用的前台 workspace，也不增加 kernel launch。eager 同样保留真实补载结果作为该步预取基线。命中块保留在候选集合中以保护其驻留槽；只从有效搬运列表排除，避免去重后反而被 LRU 淘汰。V4 的 PlanOnly 仍在真实 swap-in 完成后读取 hot table，并新增完全重叠预测的 CUDA 回归。

**copy kernel 的调用次数不等于实际搬运次数。** full graph 即使 `miss_count=0` 也会重放 copy 节点，但不访问 host KV payload。通用路径的实际搬运字节应分别按补载/预取的 `sum(miss_count) * block_size * storage_dim * dtype_bytes` 统计；比较 block ID 时还必须匹配请求、层、head 和 step。预取与真实集合不同导致的跨步驱逐/再补载，需要单独统计，不能由同一步去重保证消除。

无需 checkpoint 的去重验证：

```bash
# CPU 数值与实际 Triton planner/copy 内核解释执行
TRITON_INTERPRET=1 SGLANG_USE_CPU_ENGINE=0 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_oasiskv_dedup.py
# NVIDIA CUDA：真实双图/侧流依赖及 V4 hot-table 去重
PYTHONPATH=python python -m pytest -q \
  test/registered/cuda_graph/test_oasiskv_cuda_graph.py \
  test/registered/cuda_graph/test_oasiskv_v4_cuda_graph.py
```

### Decode 提交元数据与同步

`build_oasiskv_commit` 只构造固定的 root-only 提交；token 仍由此前的 target
`sample()` 决定。target runner 在初始化时预分配只读的 `[0, 2, ...]` 根行索引和
全 1 的 `accept_lens`，decode 按实际请求数取视图，不将 graph padding 提交给调度器。
eager/verify/capture 共用 request-major 索引构造函数；生产 CUDA 路径仅检查形状、
dtype、device 和容量，不执行 `torch.equal`、标量回读或每步 `ones/arange` 分配。

CPU 路径继续检查行索引内容；设置 `SGLANG_DEBUG_OASISKV_COMMIT=1` 可在 GPU 上用
`torch._assert_async` 检查内容。它不会让 CPU 等待，但会增加检查 kernel；异常在后续
同步点报告，不能当作可恢复的 Python `ValueError`。正式性能测量保持默认关闭。

EAGLE draft-extend 的 plan stream 显式等待主流生成采样 token、根行特征和输入转换，
不依赖提交检查的隐式 CPU 同步。V4 C4 提交使用 CPU 已知的 `[1] * B`，保留 selection
和 rollback 的 event 依赖；普通 speculative decoding 仍读取实际接受长度。
此优化不改变采样、随机数调用或 draft 永不接受的语义，也不代表整个 decode 已无其它同步。

无需 checkpoint 的回归（CUDA 环境额外执行跨流初始化、无主机同步检查及 graph replay）：

```bash
SGLANG_USE_CPU_ENGINE=0 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/spec/test_oasiskv_commit.py \
  test/registered/unit/spec/test_oasiskv_lookahead.py \
  test/registered/unit/spec/test_oasiskv_dspark.py \
  test/registered/unit/managers/test_hisparse_dspark.py \
  test/registered/unit/managers/test_hisparse_eagle.py \
  test/registered/unit/model_executor/runner/test_oasiskv_graph_inputs.py
```

性能验证应同时检查 CPU 同步、target 到 draft-extend 的 GPU 空档和稳态 TPOT/TPS。
`torch.equal` 的 CPU 耗时包含此前排队的 GPU 工作，不能直接视为可节省的 TPOT。
本地无 CUDA GPU，尚未验证 GPU 输出逐 token 一致性或实际加速幅度。

## 准备与执行

在安装了本分支 SGLang、CUDA、FA3 和 sgl-kernel 的目标机器执行。额外客户端依赖：

```bash
python -m pip install -r benchmark/oasiskv/requirements.txt
```

两个 target 和 draft 均使用上表中的本地目录，已写入 `config.json`。
论文列出的 draft 来源分别为 `Tengyunw/qwen3_8b_eagle3` 和 `nvidia/Qwen3-235B-A22B-Eagle3`，
供核对 checkpoint 版本。

运行前会检查本地目录中的配置文件及权重 shard，记录解析后的路径、配置文件哈希和权重文件元数据。
benchmark 不会下载模型；本地目录缺失时 preflight 直接报错。
正式复跑应保留同一份 checkpoint，并记录其来源 revision。

```bash
# 不需要 GPU、权重或数据即可检查 144 点矩阵及每点完整命令。
python benchmark/oasiskv/run.py plan > /tmp/oasiskv-plan.json

# 需要本地数据和 tokenizer，不需要 GPU。输出约 257 MiB token 数组（两个模型）。
# 只为指定模型准备时加 --models qwen3-8b。
python benchmark/oasiskv/prepare.py --output /tmp/oasiskv-data

# 检查资源、依赖、GPU、checkpoint、数据哈希；缺失时返回非零。
python benchmark/oasiskv/run.py preflight --data /tmp/oasiskv-data \
  --output /tmp/oasiskv-preflight

# 每个方法先执行两个短输出请求，单并发；独立目录，不混入正式结果。
python benchmark/oasiskv/run.py run --smoke --models qwen3-8b \
  --data /tmp/oasiskv-data --output /tmp/oasiskv-smoke-8b
python benchmark/oasiskv/run.py run --smoke --models qwen3-235b-a22b \
  --data /tmp/oasiskv-data --output /tmp/oasiskv-smoke-235b

# 正式矩阵；也可以用 --models / --methods 分批运行。
# 使用不同过滤条件时分开结果目录，避免混合不同 run fingerprint。
python benchmark/oasiskv/run.py run --data /tmp/oasiskv-data \
  --output /tmp/oasiskv-fig11

# 中断后跳过成功点、重跑失败点，原失败目录归档为 .attempt-*。
python benchmark/oasiskv/run.py run --resume --data /tmp/oasiskv-data \
  --output /tmp/oasiskv-fig11

python benchmark/oasiskv/report.py /tmp/oasiskv-fig11 --plot
python benchmark/oasiskv/report.py /tmp/oasiskv-fig11 --plot --tps-field decode_window_tps
```

可将配置复制到结果目录以外的位置修改，用所有命令的 `--config PATH` 指向它。
`prepare` 不覆盖已有 token 文件；修改数据、tokenizer 或 seed 后使用新数据目录。
运行采用 closed-loop：请求完成后立即补发，直到 1,024 请求结束；没有人为请求速率上限。
每次启动使用独立进程组，退出时只清理自己创建的进程。服务端仅绑定 localhost，端口被占用即报错。
默认请求超时 2 小时、启动 1 小时、单点测量 24 小时，可在配置中修改。

## 指标定义

| 字段 | 定义与限制 |
| --- | --- |
| `output_tps`（默认图） | 成功请求的实际输出 token 总量 / 整个测量阶段 wall time，包含排队、prefill 和尾部排空 |
| `decode_window_tps` | Σ(每请求输出 token − 1) / (全局最后 token 到达 − 全局第一 token 到达)；剔除每请求首 token，仍包含后续 prefill、排队与排空，不是纯 GPU decode kernel TPS |
| `running_batch_mean` | 测量期服务端 FPM 中 `num_decode_requests > 0` 的记录按 forward 样本求平均；同时报告 p50、max 和样本数；不是客户端并发，也不是 GPU 时间加权平均 |
| `tpot_mean_ms` | 每请求 `(最后 token 到达 − 首 token 到达)/(实际输出 token 数 − 1)`，再对请求求平均；另报 p50、p95 |
| `ttft_mean_ms` | 从实际发起 HTTP 请求到首 token 到达的均值，包含服务端排队和 prefill |

论文未完整披露 TPS 的计时边界、请求总量和 running batch 的平均方式。
因此同时保留两种 TPS，不能直接把任一种声称为论文内部统计口径。
所有客户端时间采用同一 monotonic clock，记录原始请求级时间和服务端 token 数，
不通过输出文本重新分词、不把 draft token 计入有效输出。
若首次 streaming chunk 已包含多个 token，就无法准确恢复第一个 token 的时间；该点标为 invalid。

FPM 订阅 TP rank 0 的一个 IPC endpoint，避免将八个 TP rank 相加。
本分支修正 `TARGET_VERIFY` 被误归为 prefill 的统计，按每个 request 的 sequence length 计数，
不按 root/draft 展开的 query 行数计数。warmup 后等到 publisher idle heartbeat 再切入测量，
测量结束也先排空遥测。只统计实际发布的 forward 样本：现有 FPM 可能跳过尚未获得 GPU timing 的迭代，
publisher 队列满时也可能丢样本，故该均值不是无损的每步审计。检测到 ZMQ sequence gap 或无 decode 样本，
该点标 invalid；生产者入队前的丢样本不一定能从 sequence gap 检出。
FPM 有 CUDA event 与 IPC 开销，两方法采用一致配置；正式报告前应另测关闭遥测的开销。

## 产物与验收

- `run.json`：配置、代码身份、解析后的 checkpoint/data 身份与 resume fingerprint。
- `environment.json`：软件版本、Git 状态、GPU/CPU/NUMA/PCIe 拓扑、主机内存。
- 每点 `launch.json`、`server_info.json`、`server.log`、`warmup.json`。
- 每点 `requests.jsonl`：请求级 token 数、首末 token 时间、结束原因、错误。
- 每点 `forward.jsonl`：原始 FPM（包含 phase、sequence ID 和接收时间）。高吞吐长测会产生较大日志，预留磁盘。
- 每点 `result.json`：成功、失败或 invalid；不将 OOM 绘制为 TPS=0。
- `points.csv`、`summary.csv`：逐次结果与有效重复的均值、样本标准差、有效重复数。
- `fig11-*.png/pdf`：两模型 × 两长度，TPS / running batch / TPOT 三行；失败点留空。

CPU 自测（包含本地伪 HTTP/SSE 和 ZMQ 数据，不产生真实性能结果）：

```bash
python -m unittest discover -s benchmark/oasiskv -p 'test_*.py' -v
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 python -m pytest -q \
  test/registered/unit/observability/test_forward_pass_metrics.py
```

正式接受性能点前，应确认：两个模型两方法 smoke 均通过；输出严格 2,048；
服务端实际 backend/TP/graph 配置符合预期；OasisKV 没有接受 draft；遥测完整性检查通过；
每点至少三次成功重复且没有其他 GPU 作业干扰。运行器保留证据，不自动宣称这些检查已在 GPU 上完成。

## UVA copy kernel 对比

通用 OasisKV full graph 的真实 query 补取与预测预取共用 tiled copy kernel，
但 launch 预算独立：`refill_ctas_per_sm=8`、`prefetch_ctas_per_sm=1`。
两项都是整个 grid 的 CTA 总预算除以 SM 数，不表示每个 SM 必定驻留多少 CTA，
也不隔离 PCIe、HBM 或 L2 带宽。batch 增大时，CTA 循环处理多个 request/head，
不会因为组数增多而突破预算；所有已选 miss 都会搬完。
H100（132 SM）、捕获 batch=8、8 KV heads 时，默认补取 1,024 CTA、预取 128 CTA；
原实现两者都是 256 CTA。每 CTA 为 128 threads，payload 按约 2 KiB tile 分配。
实际活跃 CTA 还取决于 miss 数和 padding，不能由 grid 推导搬运量。

以下在 NVIDIA CUDA GPU 上执行，不需要加载模型。源数据是真实 pinned host memory，
目的地是 GPU residency buffer；稳定地址的 graph 在不同 Counts 下重复重放。
默认验证结果逐字节相同，包含空闲槽位与未覆盖区域，并输出实际 miss 数、payload 字节数、
copy 延迟 p50/p95 和 GB/s。每个计时样本前用 128 MiB working set 驱逐 L2，
驱逐不计入 copy 时间；可按 GPU L2 容量调整 `--flush-mib`。
在与服务相同的 GPU、CPU/NUMA 绑定下运行，避免改变 host memory 的物理位置。

```bash
# 标准 Qwen3-8B：K/V head_dim 各 128，storage=256；7 live、捕获 batch=8。
# 扫描真实每 head miss 数；同时比较原 kernel、新补取与新预取预算。
PYTHONPATH=python python benchmark/oasiskv/bench_copy.py \
  --batch 8 --live-batch 7 --heads 8 --storage 256 \
  --misses-per-head 0,1,8,32,128 --output /tmp/oasiskv-copy-256.json

# 复核「总共 64 次 miss、storage=960」的假设，不把 64 个 grid 组当成 64 次 miss。
PYTHONPATH=python python benchmark/oasiskv/bench_copy.py \
  --storage 960 --total-misses 64 --output /tmp/oasiskv-copy-960.json

# 预取与合成 GEMM 并发，扫描预取预算，观察 compute_slowdown 与 joint.p50_ms。
# 调整 matmul-size，让 compute_alone.p50_ms 接近所关心的主流计算窗口。
PYTHONPATH=python python benchmark/oasiskv/bench_copy.py \
  --misses-per-head 1,8,32 --matmul-size 2048 \
  --prefetch-ctas-per-sm 1,2,4 --output /tmp/oasiskv-copy-overlap.json
```

`payload_bytes = sum(有效 Counts) × block_size × storage × dtype 字节数`。
BF16、block=16、storage=256 时每 miss 为 8 KiB，storage=960 时为 30 KiB。
`compute_slowdown` 是并发 GEMM 相对单独执行的耗时比；`joint` 包含两条流完成的总时间。
GEMM 只用于对比资源干扰，不能替代真实 attention/MLP trace、TPS 或 TPOT。
脚本只计 copy，不包含 plan selection、publish 或整模型的等待。

整服务 A/B 可在原 `--hisparse-config` JSON 中加入以下字段（保留原 top_k 等配置）：

```json
{"copy_kernel":"tiled","refill_ctas_per_sm":8,"prefetch_ctas_per_sm":1}
```

将 `copy_kernel` 设为 `legacy`，可恢复原 copy 算法和固定 4 workers/request/head，
此时两项 CTA 预算不生效。修改预算或 kernel 后重启服务以重新捕获 CUDA graph。
本开关只作用于通用 full-graph UVA copy；通用 eager 暂不读取这些调优项，DeepSeek-V4 仍采用其 C++ 搬运实现。
真实补取仍在 attention 前完成，预取写入仍依赖本层 attention 读取结束，未改变 stream/event 依赖。
优先按真实主流耗时、联合耗时与最终 TPS/TPOT 选择预算，不能只按 copy 单独运行的带宽选择。

## HBM 稀疏 attention kernel 对比

通用路径默认 `attention_kernel=auto`：FP16/BF16、GQA group ≥ 2、K/V head dim
各 ≤ 256、block_size 为不超过 64 的 2 的幂时使用 physical-page grouped kernel。
MHA、FP32、大维 MLA 和不支持的 block geometry 保留原 SIMT kernel；
DeepSeek-V4 继续使用其原生 FlashMLA。这不是直接移植 V4 的共享 K/V、FP8 数据布局。

优化包含：

- 将物理页地址和有效长度合入前台 `publish_plan`，与真实 query 补载后的状态一致。
  descriptor 地址固定，每次 replay 重写，padding 行清零，支持请求重排和页复用。
  eager 使用独立 metadata kernel。metadata 只含地址和长度，不复制 KV。
- 每 CTA 处理同一 KV head 的多个 query heads，以及 real/draft 两行。
  K/V tile 在组内共享；QK/PV 使用 Tensor Core，softmax 和累加使用 FP32。
  对齐的 K/V 特征维使用向量读取，避免同时持有展开后的 FP32 K、V。
- 历史块、active block、draft token 分开处理；draft KV 仅对 draft 行可见。
  split-KV 根据 batch、head 分组及 SM 数静态选择，所有 workspace 在 capture 前分配。
  workspace 在相同 head geometry 的层之间共享，不按最大 batch 一律分配 32 个 splits。

选择的 KV 集合、真实 token 提交规则、前台补载与后台预取的依赖保持原语义。
没有增加 HBM staging buffer，也没有放宽预取写入必须等待 attention 读取结束的约束。
浮点计算顺序和概率矩阵精度发生变化，数值按 FP16/BF16 容差验证，不保证逐 bit 相同。

同一启动配置中加入 `"attention_kernel":"legacy"` 可做整服务 A/B；
改回 `"attention_kernel":"auto"` 恢复自动选择。每次都重启服务并重新捕获 graph。
不要同时改 top_k、max_new_blocks、copy_kernel 或请求并发，以免混入其他因素。

HBM-only 微基准无需 checkpoint；所有 KV 已经驻留在 HBM，新旧实现读取完全相同的
物理页、query 和 mask。脚本先校验结果，再用 CUDA graph 计时，并分别报告
`metadata`、`attention_and_reduce`、`metadata_attention_reduce` 的 p50/p95，
以及寄存器、spill、shared memory 和有效 KV 字节数。默认在每个样本前驱逐 L2，
驱逐操作不计时；`--flush-mib 0` 可测 warm-cache 场景。

```bash
# Qwen3-8B 对应的本地 head geometry；包含 padding 的 batch=8 / live=7。
PYTHONPATH=python python benchmark/oasiskv/bench_attention.py \
  --batch 8 --padding 1 --heads 8 --group 4 \
  --key-dim 128 --value-dim 128 --top-k 2048 --block-size 16 \
  --output /tmp/oasiskv-attention-b8.json

# 扫描 batch，以及 CTA head 分组、split 数和 KV tile 大小。
# 默认只比较 legacy 和 auto；指定以下选项才额外编译 sweep 配置。
PYTHONPATH=python python benchmark/oasiskv/bench_attention.py \
  --batch 1,8,32,128,256 --head-tiles 2,4,8 --splits 4,8,16 --tiles 32,64 \
  --output /tmp/oasiskv-attention-sweep.json

# CUDA 数值验证，以及完整模型前的请求重排、padding、回收复用、异步 graph 验证。
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 python -m pytest -q \
  test/registered/unit/mem_cache/test_oasiskv_attention.py \
  test/registered/cuda_graph/test_oasiskv_cuda_graph.py
```

`useful_GBps` 的分子按实际 live request、KV head、有效 token 和 K/V 维度计算一次，
不乘 GQA query heads 或 real/draft 行数；它不是 Nsight 的 DRAM 流量计数。
判断带宽时还需采集 DRAM/L2 bytes、sector 利用率、memory stalls 和实际 occupancy。
最终以 `metadata_attention_reduce` 与整服务 TPS/TPOT 为准，不能只看 partial kernel。
不以离线编译出的寄存器数宣称 GPU 加速：当前交付环境无 GPU，CUDA replay、
BF16 数值误差、真实性能及最终 token 质量需在 H100 上按上述命令确认。
CPU `TRITON_INTERPRET=1` 可检查 FP16 数值、寻址和 mask；Triton 3.7 的 BF16 dot
解释器将位模式当 uint16 处理，因此这些 BF16 用例只在 CUDA 上运行。
