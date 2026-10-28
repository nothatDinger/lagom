# 用 SGLang 原生 bench 测 OasisKV

适用分支：`perf/oasiskv`。使用 `run.py --client sglang-bench`，runner 启动并校验服务，
请求由 **`python -m sglang.benchmark.serving`** 子进程发送，TPS/TPOT 直接取其结果；
实际 running batch 由服务端 Forward Pass Metrics（FPM）采集。
本仓库 `python -m sglang.bench_serving` 是同一入口的兼容别名，会提示弃用。

正式结果标为 `fig11-sglang-bench`。不指定 `--client` 时仍使用原来的 custom client。
两种协议的输入构造和流式计时边界不同，不能将它们的点拼成同一条对照曲线。
本方案不改变 server 或 OasisKV 的执行路径。

## GPU 执行 agent 的顺序

从仓库根目录执行，使用安装了本分支 SGLang/CUDA 依赖的 Python 环境。
先确认分支并记录 `git rev-parse HEAD`，确认选用 GPU 没有其他作业。
设备由 `config.json` 中各模型的 `cuda_visible_devices` 决定，runner 会覆盖外层同名环境变量。
变更设备或请求数时复制配置文件，并在以下所有命令中传同一个 `--config`。

```bash
python -m pip install -r benchmark/oasiskv/requirements.txt
python -m sglang.benchmark.serving --help

# 只打印 server、warmup 和正式 bench 命令，不加载模型。
python benchmark/oasiskv/run.py plan --client sglang-bench \
  --models qwen3-8b --concurrencies 8 16 --input-lengths 16384

# 检查本地 ShareGPT、模型文件、tokenizer、H100 数量与容量。
# 原生方案无需 prepare.py，也不读取 --data/prompts.npy。
python benchmark/oasiskv/run.py preflight --client sglang-bench \
  --models qwen3-8b --output /srv/oasiskv/bench-preflight-8b

# 两种方法各一次启动检查：固定C=1、2请求、输出32，不能当C=8性能点。
python benchmark/oasiskv/run.py run --client sglang-bench \
  --models qwen3-8b --smoke --output /srv/oasiskv/bench-smoke-8b

# 正式8/16档：每点1024请求、输出2048、3次重复，dense和oasiskv均测。
python benchmark/oasiskv/run.py run --client sglang-bench \
  --models qwen3-8b --concurrencies 8 16 --input-lengths 16384 \
  --output /srv/oasiskv/bench-8b-c8-c16-16k
python benchmark/oasiskv/report.py /srv/oasiskv/bench-8b-c8-c16-16k --plot
```

235B 使用 **8 × H100 80GB，TP=8**，8B 使用 **1 × H100 80GB，TP=1**。
235B draft 路径为 `/mnt/public_data/Qwen3-235B-A22B-Eagle3`，注意大小写：

```bash
python benchmark/oasiskv/run.py preflight --client sglang-bench \
  --models qwen3-235b-a22b --output /srv/oasiskv/bench-preflight-235b
python benchmark/oasiskv/run.py run --client sglang-bench \
  --models qwen3-235b-a22b --smoke --output /srv/oasiskv/bench-smoke-235b
python benchmark/oasiskv/run.py run --client sglang-bench \
  --models qwen3-235b-a22b --concurrencies 8 16 --input-lengths 16384 \
  --output /srv/oasiskv/bench-235b-c8-c16-16k
python benchmark/oasiskv/report.py /srv/oasiskv/bench-235b-c8-c16-16k --plot
```

小档验证后，去掉两个筛选参数可跑完整矩阵；新协议/筛选组合使用独立输出目录：

```bash
# 2模型 × 2方法 × 2输入长度 × 6最大并发 × 3重复 = 144次独立服务启动。
python benchmark/oasiskv/run.py run --client sglang-bench \
  --output /srv/oasiskv/bench-full
python benchmark/oasiskv/report.py /srv/oasiskv/bench-full --plot
```

完整矩阵：输入16384/32768、输出2048、C=8/16/32/64/128/256。
可用 `--methods oasiskv` 单独调试；正式对比需要同一协议的 dense 与 OasisKV。
恢复任务时使用完全相同的命令并加 `--resume`。配置、筛选点、代码或依赖变化后使用新目录。
OOM、启动失败或超时保留为失败点，不自动降低并发、输入长度或 KV 预算。

## server 与 bench 如何配套

| 项目 | C=8 | C=16 | 含义 |
| --- | --- | --- | --- |
| server `--max-running-requests` | 8 | 16 | 全局调度请求数上限 |
| server decode graph `max_bs` | 8 | 16 | 捕获容量；实际 replay bucket 看日志 |
| bench `--max-concurrency` | 8 | 16 | 最大在途请求数 |
| bench `--num-prompts` | 1024 | 1024 | 全部请求数，不是 batch size |
| bench `--request-rate` | inf | inf | 由并发限制驱动，无额外到达速率限流 |
| 实际 running batch | FPM测量 | FPM测量 | 可能低于C，不能从配置推定 |

TP=8 时 C=8 仍是 **8个请求**，不乘以8。paired verify 的 `2B` query 行和graph padding bucket
都不是实际请求 batch。Fig.11 横轴按 Max concurrency 扫描，不承诺每个 decode forward 都保持固定 batch。
详见 [CONCURRENCY.md](CONCURRENCY.md)。

下面是手工检查单点的命令。终端A启动8B OasisKV，终端B在 `/health` 就绪后运行原生 bench。
手工方式没有 runner 的 FPM 订阅器和结果校验；完整三指标测试用上面的自动入口。

```bash
# 终端A。C=16时同时改max-running-requests、max_bs，以及终端B的max-concurrency。
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /mnt/public_data/Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 30000 --tp 1 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --context-length 40960 \
  --max-running-requests 8 \
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":8}}' \
  --mem-fraction-static 0.9 --chunked-prefill-size 8192 \
  --disable-radix-cache --random-seed 42 --stream-interval 1 \
  --attention-backend oasiskv \
  --speculative-draft-model-path /mnt/public_data/Qwen3-8B_eagle3 \
  --hisparse-config '{"top_k":2048,"block_size":16,"prefill_chunk_size":256,"max_new_blocks":null}'

# 终端B。此手工示例用内置warmup；自动runner另起warmup进程隔离FPM。
python -m sglang.benchmark.serving \
  --backend sglang --host 127.0.0.1 --port 30000 \
  --model /mnt/public_data/Qwen/Qwen3-8B \
  --tokenizer /mnt/public_data/Qwen/Qwen3-8B \
  --dataset-name random \
  --dataset-path /home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json \
  --random-input-len 16384 --random-output-len 2048 \
  --random-range-ratio 1 --tokenize-prompt \
  --num-prompts 1024 --request-rate inf --max-concurrency 8 \
  --seed 42 --temperature 0 --warmup-requests 1 \
  --output-file /tmp/bench-8b-oasiskv-c8-16k.jsonl --output-details
```

dense 对照将 server backend 改为 `fa3`，移除 draft 和 hisparse 参数，其余保持一致；
bench 参数一致，输出文件换新名字。原生 bench 默认向已有文件追加，不要重复使用同一文件。
自动 runner 会拒绝复用已有单点输出，并保存失败尝试。

## 数据和计时口径

固定长度 ShareGPT 输入使用 `--dataset-name random --dataset-path ... --random-range-ratio 1 --tokenize-prompt`。
本仓库 `random` 按 seed 打乱至少有两轮的 ShareGPT 对话，取首轮 prompt，按 target tokenizer 编码，
重复或截断 token IDs 到指定长度，直接发送IDs；tokenizer默认添加的特殊token如有也计入长度。
`random-ids` 是随机整数token，不是本方案。`sharegpt` 模式保留自然输入长度，
`--sharegpt-context-len` 只是过滤上限；不能用它加 `--random-input-len` 宣称已固定16K/32K。

同一配置/seed下，两种方法使用相同输入；16K是对应32K输入的前缀。
它不是 custom client 的“多轮串接后不重叠切片”，也不用于质量评估。
runner 校验本地JSON并记录源文件/tokenizer指纹，防止静默使用下载的数据。
如果 tokenizer 过滤后实际请求数不足，结果校验会拒绝该点。

每点独立启动server，完成2个输出32 token的串行warmup，等待FPM idle heartbeat；
正式bench另起子进程，内置 `--warmup-requests 0`，测完后再次等idle。
warmup复用相同seed下的前两个prompt，服务端关闭radix cache，不沿用custom路径的独立warmup切片。
加载模型/tokenizer、数据准备及warmup不进入原生bench测量计时。
原生请求超时目前为6小时；runner的 `point_timeout_s`（默认24小时）限制每次bench进程，
`request_timeout_s`不会改写原生bench内部超时。超时时只清理本测试创建的进程组。

| 汇总字段 | 来源/公式 |
| --- | --- |
| `output_tps` | 原生`output_throughput`：成功请求completion token数 / benchmark duration；包括prefill、decode、尾部排空及计时范围内的server_info查询 |
| `tpot_mean_ms` / p50 / p95 | 原生TPOT：各成功请求 `(latency - TTFT) / (output_tokens - 1)` 的统计 |
| `running_batch_mean` / p50 / max | TP rank0 FPM `scheduled_requests.num_decode_requests > 0` 的forward样本统计，含升降载；按forward计，不按时间加权 |
| `decode_window_tps` | 原生未提供必要边界，保留null；原生方案绘图使用`output_tps` |

不使用 `total_output_tokens_retokenized` 计算TPS，不将draft token加入输出吞吐。
原生 `concurrency` / `max_concurrent_requests` 是客户端统计，不是服务端running batch。
原生SSE以**首个非空text**记录TTFT，与custom client按completion_tokens增量的边界有区别；
空文本token和合并chunk会影响该边界，`stream_interval=1`不保证逐token可见text。
本方案保留原生指标定义，不声称与custom路径严格相等。

## 验收与交付

每个单点目录应有：

- `launch.json`、`server_info.json`、`server.log`：真实server参数和日志。
- `bench-launch.json`、`bench-warmup-launch.json`：实际原生CLI命令。
- `bench.jsonl`、`bench-warmup.jsonl`及对应`.log`：原生结果、逐请求长度/错误/时延明细。
- `forward.jsonl`：带warmup/measurement标记的rank0 FPM。
- `result.json`：归一化指标及失败原因。根目录还有配置/代码/环境指纹。

只有 `status=ok` 的点进入重复均值。runner检查完成请求数=1024、输入/输出长度准确、
无请求错误、有流式时延样本、FPM无缺口、实际batch不超过C。
启动校验要求full decode graph覆盖C，OasisKV为lookahead模式、draft宽度2；
日志中还应核对实际capture/replay。平均accept length若大于1会拒绝该点；缺失则记录
`acceptance_telemetry=unavailable`，不能声称已逐请求验证强制拒绝。
原生明细不导出每请求finish_reason、服务端prompt_tokens和首chunk token数，校验范围相应有限。

交付 `summary.csv`、`points.csv`、三指标图和所有失败点的日志。
报告C=8/16时同时给出观测running batch，不写成“固定batch=8/16”。
本工作区仅做CPU协议验证（含真实bench CLI对接模拟SSE服务）；GPU性能和模型启动由执行agent验证。
