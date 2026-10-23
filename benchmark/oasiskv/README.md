# OasisKV 性能测试准备：Fig.11

本目录基于 `oasiskv` 的提交 `333551cf8f6884d1a9eef30dfd70932cd547d9c4`，
为 [OasisKV v1](https://arxiv.org/pdf/2608.08097v1) §5.2、§5.3.1 / Fig.11
准备长度、并发扫描与指标采集。**尚无 GPU 性能结果。**

## 测试矩阵

| 项目 | 配置 |
| --- | --- |
| Target | `/mnt/public_data/Qwen/Qwen3-8B`；`/mnt/public_data/Qwen/Qwen3-235B-A22B` |
| GPU / TP | 8B：1 × H100 80GB / TP=1；235B：8 × H100 80GB / TP=8 |
| Draft（论文所列） | `Tengyunw/qwen3_8b_eagle3`；`nvidia/Qwen3-235B-A22B-Eagle3` |
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

- 原论文使用 vLLM V1 v0.12.0；dense 使用 FA3/full CUDA graphs，OasisKV 使用 piecewise graphs。
  此处是 **SGLang 移植版与同框架 dense 的对照**。当前通用 OasisKV 禁用 CUDA graphs，预测在主流、
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

权重必须事先准备好。两个 target 使用上述本地路径；draft 可下载到 Hugging Face 缓存，
或把 `config.json` 中 `draft` 改为本地目录。下载操作独立于 benchmark：

```bash
hf download Tengyunw/qwen3_8b_eagle3
hf download nvidia/Qwen3-235B-A22B-Eagle3
```

运行前会用 `local_files_only=True` 解析已缓存 snapshot，记录解析后的路径、配置文件哈希，
检查权重索引中所有 shard 存在；不会在计时阶段下载模型。
正式复跑建议保存这些 snapshot 并使用固定 revision；无需重复下载数百 GB 的 target 权重。

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
