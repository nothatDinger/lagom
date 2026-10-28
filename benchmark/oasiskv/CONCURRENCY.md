# Fig.11 并发测试执行指南

适用分支：`perf/oasiskv`。目标是让每个实验点的 server、请求负载和结果标签一致。
使用本目录的 `run.py` 统一启动 server 和 benchmark；不要另启一个配置不匹配的 server。
如需使用 SGLang 原生 bench，见 [SGLANG_BENCH.md](SGLANG_BENCH.md)，在 runner 命令中添加
`--client sglang-bench`；下面的默认命令使用 custom client。两者保持相同的并发配置规则。

## 先确定测的是哪个实验

[原文 Fig.11 / §5.3.1](https://arxiv.org/pdf/2608.08097v1#page=9) 的横轴是 **Max concurrency**：
扫描最大并发，再分别报告 TPS、实际 running batch、TPOT。因此横轴 8/16 不承诺每轮实际 batch 恰好等于 8/16。
[Fig.2](https://arxiv.org/pdf/2608.08097v1#page=2) 的 batch 8/16/32/64 则是 roofline 模型计算出的延迟分解，
不是服务压测结果；Fig.12 是 PD 分离场景的请求到达率扫描，也不要混入本实验。

本仓库执行的是 SGLang 移植对照。以下 closed-loop、每点 1,024 请求、3 次重复和具体 server 参数
是本仓库选定的可重复协议；论文没有给出这些参数的完整启动命令。不要把它们称为作者原始脚本。

## 五个数分别代表什么

| 数量 | 8 档 | 16 档 | 配置/来源 |
| --- | --- | --- | --- |
| 客户端最大在途请求数 C | 8 | 16 | `config.concurrencies`，客户端创建 C 个异步 worker，完成一个再补一个 |
| 服务端运行请求上限 S | 8 | 16 | 本协议固定 `--max-running-requests C` |
| Decode graph 最大请求 batch G | 8 | 16 | `--cuda-graph-config '{"decode":{"backend":"full","max_bs":C}}'` |
| 实际 decode 请求数 B | 测量值，至多 8 | 测量值，至多 16 | `forward.jsonl` 的 `scheduled_requests.num_decode_requests` |
| 总请求数 N | 1,024 | 1,024 | `config.num_requests`；不是 batch，也不因 C 改变而改成 8/16 |

S 和 G 是上限，不能替代客户端负载。只把 server 上限设成 16、却逐个 `await` 请求，仍是串行测试。
本客户端是 C 个并行 worker，没有人为 QPS 限速，HTTP 连接池也不额外限制 C。
在途请求还包括排队和 prefill，因此 `B < C` 本身不意味着测试失败。
本协议每点按 C 重启服务；不要复用一个 S=256、G=256 的 server 测所有 C，
因为这会改变 graph 和 KV 池的预留规模，成为另一套配置。

OasisKV 的 target verify 每个请求有 real/draft 两行：B=8 对应 16 行，B=16 对应 32 行。
`max_bs`、`max-running-requests` 和客户端 C 都按**请求数**配置，不乘 2。
例如实际 B=7 可能重放 graph bucket=8，图内有 16 行，其中一对为 padding；这不是实际 batch=8 或 16。
只有真实输出 token 进入 TPS，lookahead 行不计入输出。

235B 的 TP=8 代表八张卡协作处理同一批请求。C=16 是整个服务实例 16 个请求，
不是每卡分配 2 个请求，也不是八个独立 server 各收 16 个请求。FPM 只统计一个发布 rank。

`--chunked-prefill-size 8192` 是 prefill token 分块量，`top_k=2048` 是每 head 的历史 KV token 预算；
两者均不是请求 batch。不要为了“达到 batch 16”修改这两个数。

## GPU 机器执行顺序

使用已经安装本分支依赖的 Python 环境，从仓库根目录执行。保留 agent 自己未提交的 harness 改动，
先记录 `git status --short`、`git rev-parse HEAD` 和差异，不要通过 hard reset 清理工作区。
结果中的 `run.json` / `environment.json` 也会记录代码和运行环境。

默认模型和设备：

| 模型 | target | draft | GPU / TP |
| --- | --- | --- | --- |
| qwen3-8b | `/mnt/public_data/Qwen/Qwen3-8B` | `/mnt/public_data/Qwen3-8B_eagle3` | `0` / 1 |
| qwen3-235b-a22b | `/mnt/public_data/Qwen/Qwen3-235B-A22B` | `/mnt/public_data/Qwen3-235B-A22B-Eagle3` | `0,1,2,3,4,5,6,7` / 8 |

`run.py` 和 preflight **用 config 中的 `cuda_visible_devices` 设置子进程环境**，会覆盖外层 shell 的同名变量。
若作业分配的是其它 GPU，复制 `config.json` 并修改对应 model 的设备列表，所有命令都加 `--config PATH`。
不要依赖只在命令前写 `CUDA_VISIBLE_DEVICES=...` 来覆盖本运行器。保持分配与实际可见设备一致。

先准备输入（已准备且通过 checksum 检查的目录可复用）：

```bash
python -m pip install -r benchmark/oasiskv/requirements.txt
python benchmark/oasiskv/prepare.py --models qwen3-8b --output /srv/oasiskv/data
python benchmark/oasiskv/run.py preflight --models qwen3-8b \
  --data /srv/oasiskv/data --output /srv/oasiskv/preflight-8b
```

默认数据源为 `/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json`。
prepare 将 ShareGPT 内容打包成固定 token 行；run 截取精确的 16,384/32,768 个 input IDs，
不以字符数、原始 ShareGPT 对话长度或重新 tokenize 的文本替代。

先检查**将要执行的 server 和客户端参数**，此操作不启动模型：

```bash
python benchmark/oasiskv/run.py plan --models qwen3-8b \
  --concurrencies 8 16 --input-lengths 16384 > /srv/oasiskv/plan-8b-c8-c16.json
```

每个 plan 点包含完整 server `command`、`cuda_visible_devices` 和 `client` 配置。
上述筛选产生 12 点：2 档并发 × 2 方法 × 3 重复；每点仍是 1,024 请求、2,048 输出。
`--concurrencies` 和 `--input-lengths` 只筛选 config 中已有的值；参数采用空格分隔，不能写 `8,16`。

启动检查与正式压测分开：

```bash
# 仅验证能启动和生成：C=1，2 请求，每请求 32 输出；不能当成 8/16 档结果。
python benchmark/oasiskv/run.py run --smoke --models qwen3-8b \
  --data /srv/oasiskv/data --output /srv/oasiskv/smoke-8b

# 正式 16K 输入，8/16 档；每点独立启动/停止 server，默认同时测 dense 和 OasisKV。
env -u CUDA_LAUNCH_BLOCKING python benchmark/oasiskv/run.py run --models qwen3-8b \
  --concurrencies 8 16 --input-lengths 16384 \
  --data /srv/oasiskv/data --output /srv/oasiskv/fig11-8b-c8-c16-i16k

python benchmark/oasiskv/report.py /srv/oasiskv/fig11-8b-c8-c16-i16k --plot
```

正式计时关闭 blocking/debug 同步和 profiler，不与其它 GPU 任务共享资源。
若只定位 OasisKV，可加 `--methods oasiskv` 并使用另一个结果目录；正式对照需两方法均有结果。
测 32K 改用 `--input-lengths 32768`，测全部档位则省略两个筛选参数。
运行器拒绝 `--smoke --concurrencies 8`，防止请求的 8 档被 smoke 静默改成 1。

235B 先为对应 tokenizer 准备数据和 smoke，再执行八卡 TP8 测试：

```bash
python benchmark/oasiskv/prepare.py --models qwen3-235b-a22b --output /srv/oasiskv/data
python benchmark/oasiskv/run.py preflight --models qwen3-235b-a22b \
  --data /srv/oasiskv/data --output /srv/oasiskv/preflight-235b
python benchmark/oasiskv/run.py run --smoke --models qwen3-235b-a22b \
  --data /srv/oasiskv/data --output /srv/oasiskv/smoke-235b
env -u CUDA_LAUNCH_BLOCKING python benchmark/oasiskv/run.py run --models qwen3-235b-a22b \
  --concurrencies 8 16 --input-lengths 16384 \
  --data /srv/oasiskv/data --output /srv/oasiskv/fig11-235b-c8-c16-i16k
```

两个模型分别使用各自 config 中的 TP 与卡数，不因 C=8/16 而调整 TP。
保留 `max_new_blocks=null`、BF16、40,960 context、greedy、ignore_eos、关闭 radix cache 和 decode full graph。
启动命令由 `common.server_command()` 生成：dense 使用 `fa3` 且不启用 draft；OasisKV 使用 `oasiskv` 和对应 EAGLE-3。

相同筛选、配置、代码和依赖才能对原目录使用 `--resume`。筛选后的 config 和完整 points 会进入 fingerprint；
更换方法、模型、并发、代码或依赖时使用新目录。不要把不同协议的结果覆盖进同一目录。

## 每点如何验收与交接

先核对 `launch.json` 的 point/argv/设备，再核对 `server_info.json` 的实际 backend、TP、
`max_running_requests`、decode graph 上限和 OasisKV lookahead 配置。检查 capture/replay 日志，
不要仅凭传入了 `max_bs` 就声称 graph 生效。运行器会检查上述解析参数；配置不符不会被接受为有效点。

从 `result.json` 和 `forward.jsonl` 读取实际 batch，禁止用 kernel grid、2B query 行数或 graph bucket 代替。
下面打印某个结果目录每点的实际 decode batch 分布，包含起步和尾部，不人为筛掉低 batch：

```bash
python - /srv/oasiskv/fig11-8b-c8-c16-i16k <<'PY'
import collections, json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob('*/result.json')):
    if '.attempt-' in path.parent.name:
        continue
    result = json.loads(path.read_text())
    hist = collections.Counter()
    forward = path.with_name('forward.jsonl')
    if forward.exists():
        with forward.open() as stream:
            for line in stream:
                row = json.loads(line)
                batch = row['scheduled_requests']['num_decode_requests']
                if row['phase'] == 'measurement' and batch > 0:
                    hist[batch] += 1
    print(path.parent.name, 'status=', result['status'],
          'C=', result['concurrency'], 'B_mean=', result.get('running_batch_mean'),
          'B_max=', result.get('running_batch_max'), 'hist=', dict(sorted(hist.items())))
PY
```

`C=8、B_mean=6.8、B_max=8` 可是正常结果；`C=8、B_max=7` 也不能单凭这一项判失败。
若 C=16 却始终 B=1，检查客户端是否确实并发、是否误用 smoke、是否大量请求处于 prefill/排队、
KV 容量或调度限制；同时查看请求 start/end 时间、服务端 queue/preemption 日志和完整 batch 分布。
不要为了画出预期 batch 曲线，降低 context、改 KV 预算或只挑 batch 满的样本。
高并发因 OOM/不支持而失败就保留失败；能运行但实际 batch 受容量限制则如实报告观测值。

交接给分析 agent 的每点记录至少包含：

- 代码 HEAD/dirty diff、模型/draft 路径、GPU/TP、方法、C、输入/输出长度、N、repeat 和配置文件。
- `status` / 失败原因、成功请求数、精确输出 token 数与 lookahead 强制拒绝检查结果。
- `output_tps`、`decode_window_tps`、`running_batch_mean/p50/max`、batch 分布、`tpot_mean_ms/p50/p95`。
- `launch.json`、`server_info.json`、`server.log`、`requests.jsonl`、`forward.jsonl` 与遥测完整性信息。

两种 TPS 的边界见主 README：前者包括整段测量的 prefill/排空；后者也不是纯 GPU decode 吞吐。
不要直接用 `C / TPOT` 替代实测 TPS，也不要把固定 B 的 attention kernel 微基准当成 Fig.11 服务测试。
若另做固定 B=8/16 的 decode 分析，应单独命名实验，并证明窗口内实际 B 固定、没有 prefill，不能据此改写 Fig.11 的测量口径。
