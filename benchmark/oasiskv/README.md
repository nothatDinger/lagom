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
真实 Top-K 补载、attention，以及独立 stream 上的选择/传输。传输等待本层 attention，
在模型末尾汇合所有侧流，因此可与后续层重叠，但不会跨过本轮 graph 尾部。
host 分配/扩容和请求回收在图外，变长 dense prefill 暂时仍为 eager。
`--disable-cuda-graph` 保留为显式 eager 对照；正式测试须确认日志中的实际 capture/replay。
性能脚本检查服务端解析后的 decode graph 配置；若关闭 graph 或最大 batch 不覆盖该点并发，会将该点判为失败。

先在目标 CUDA 环境从仓库根目录执行无需 checkpoint 的图回归测试，再进行服务 smoke：

```bash
PYTHONPATH=python python -m pytest -q test/registered/cuda_graph/test_oasiskv_cuda_graph.py
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
  当前通用路径会按 `max-running-requests` 预留 device working set，因此高档并发也可能在启动阶段容量不足；
  不能假定它一定像论文曲线那样自动维持较低的 running batch。服务端报告的实际请求上限和 KV 容量会留存。

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
本开关只作用于通用 full-graph UVA copy；通用 eager 与 DeepSeek-V4 的 C++ 搬运实现未改动。
真实补取仍在 attention 前完成，预取写入仍依赖本层 attention 读取结束，未改变 stream/event 依赖。
优先按真实主流耗时、联合耗时与最终 TPS/TPOT 选择预算，不能只按 copy 单独运行的带宽选择。
