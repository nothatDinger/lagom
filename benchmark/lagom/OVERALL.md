# LongBench-v2 overall：8×H100，Mooncake P TP4 / D TP4

入口：`run_overall.py`。指标为端到端 serving 的 output TPS 和 TPOT。
本方案与 ShareGPT 短测并存，不改变 `run.py` / `run_concurrent.py` 的模式含义。

**当前代码限制：** 编写脚本时，远端 `perf/lagom` 的
`python/sglang/srt/arg_groups/speculative_hook.py::_handle_lagom_mode` 仍明确拒绝 PD 分离：

```text
Lagom currently requires colocated prefill and decode.
```

当前 checkout 可以生成全部启动计划，但不能直接完成 Lagom 的 PD 实验。
实际执行包含 Lagom 的矩阵时，脚本在启动任何服务前报出该限制。
先由执行环境 agent 补齐并验证 PD 支持；**仅删除检查不等于完成支持**。
本提交只组织实验，不修改保护或推测解码运行时。本环境无 Python/CUDA 执行能力，
以下 CPU 测试、启动命令及 GPU 性能结果均需在执行环境验证。

## 实验口径

| 方法名 | P 节点 | D 节点 | Lagom 预取 |
|---|---|---|---|
| `hisparse` | 普通 target prefill | HiSparse，普通 decode，无 DSpark | 关 |
| `dhisparse` | target prefill + DSpark context KV 注入 | DSpark + HiSparse，整窗口 oneshot verify | 关 |
| `lagom` | 与 DHisparse 相同 | DSpark + HiSparse + 预测窗口 | 开 |

这里 Hisparse baseline **没有推测解码**。旧短测 `--modes hisparse` 是 DSpark + HiSparse，
等价于本实验 DHisparse；不要混用结果标签。

- 模型：DeepSeek V4 Flash + 配套 DSpark；模型路径在 JSON 配置中修改。
- GPU：8 张 H100；P 使用可见 GPU 0–3，D 使用 4–7；各 TP4，DP/PP/CP/DCP=1。
- 数据：LongBench-v2；精确输入 **32768 / 65536 / 131072 token**，
  包含问题、选项及 special-token wrapper。
- 输出：正式请求固定 **1024 token**，temperature=0、ignore_eos=true、streaming=true。
- 允许的最大并发：**8 / 16 / 32 / 64 / 128 / 256**。同时设为 client concurrency 和
  server max-running-requests 上限；不要求实际 running decode batch 达到该数，
  不剔除小 batch 或收尾阶段。
- 每格正式请求 **N = 4 × C**，分别为 32 / 64 / 128 / 256 / 512 / 1024。
  SGLang 另执行 1 个同输入长度、最多输出 32 token 的预热请求，不计入 N。
- 顺序：Hisparse 全部 18 格 → DHisparse 全部 18 格 → Lagom 全部 18 格；
  每组按输入长度、并发递增。共 **54 格**，每格重新启动 P、D、router。
- 每格上限 **21600 秒（6h）**，从启动第一项服务之前计时，包含服务启动、
  graph capture、输入加载、预热、正式请求及清理。预留 20 秒清理，
  软终止后强制结束整个进程组。启动阶段另有共享的 3600 秒上限，
  可在配置中调大，但仍受 6h 总预算约束。
- 数据集下载/token bank 准备为一次性工作，放在矩阵运行之前。
  整套矩阵没有 6h 总上限。
- 不自动减少请求数、输出长度、并发上限，不自动缩小 buffer 或关闭 CUDA Graph。
  失败/超时标为无效结果，其余配置继续执行。

## 控制变量和容量

`overall.example.json` 默认：

| 配置 | 值 |
|---|---|
| W / P | 7 / 7；Hisparse 无 speculative window |
| `device_buffer_size` | **8192，三组相同** |
| `host_to_device_ratio` | **2，三组相同**，沿用此前实验，需评估主机 RAM |
| `force_h2d_each_step` | false |
| target / draft MoE backend | marlin / marlin |
| `mem_fraction_static` | 0.8 |
| page / chunked prefill | 256 / 8192 |
| decode extra request slots | 显式 0，避免随 C 改变默认额外槽位数 |
| radix cache | P/D 均关闭 |
| ragged verify | `SGLANG_RAGGED_VERIFY_MODE=static` |
| decode CUDA Graph | 显式 full，1、2、4……C 的 batch size；保留 DSpark graph 路径 |
| trace、模拟接受率、跳过 KV IO | 正式测量不使用 |

DHisparse 的 target verify 为 W+1=8 行，包含 bonus token；
Lagom 增加 P=7 个预测行，逻辑窗口 W+P+1=15 行。K=512 时 union 上界 7680，
满足 `(W+P+1)*K <= min(hot,8192)`。
8192 是每请求的 hot C4 KV 槽位，不是原始输入 token 数。32k 点可能接近完整 C4 KV
常驻，可作为小缺失率对照；64k/128k 更能体现预取的潜在价值。

H100 对 native FP4 routed experts 显式选 Marlin，避免 FP8 配置自动落到 Triton 后
触发 hidden-size mismatch。不要通过禁止 FP4 experts 改变 checkpoint 权重格式。
本地需确认 target/draft 的 Marlin 和 attention backend 支持。

C=128/256、128k 不保证在任意 H100 显存和主机 RAM 配置下都能启动。
host pool、SWA/索引 KV、request slots、DSpark workspace、多 batch graph 都占用内存。
**并发上限不等于常驻请求数**；若调度器因容量只能运行较小 batch，
仍是有效的该并发上限 serving 结果。若启动 OOM 或请求失败，保留 failed 记录，
不写为 0 TPS 或成功数据。若调 buffer、host ratio、memory fraction，应三组统一修改，
并用新输出目录重跑；不能只给 Lagom 增大 buffer 或只降低某组并发。

## 精确输入

原生 SGLang LongBench-v2 loader 会过滤过长样本，不保证精确长度和 1024 个请求。
`longbench_v2_fixed.py` 先准备三组共用的输入：

1. 加载 `THUDM/LongBench-v2` train（或本地原始 JSON/JSONL/parquet），按 seed=42 打乱。
2. 分别编码 context 和原始 question + A/B/C/D + Answer:，仅裁剪 context 头/尾。
   使用 tokenizer 的 `build_inputs_with_special_tokens` 包装，不套 chat template，
   不补随机 token，不拼接不相关文档。
3. 只选 context 足够长的样本；每个长度最多保留 1024 个不同 source 样本。
   不足 1024 时按固定种子打乱并循环重复；三组、各 C 共用同一序列的前 N 项。
   若某长度没有足够长的原始 context，则明确失败。
4. 保存 little-endian uint32 token bank、每项 SHA256、source id、
   原始 context token 数、tokenizer vocabulary hash、dataset fingerprint、请求顺序。
   重复请求不重复落盘长 prompt。
5. benchmark 适配器只替换当前进程的 dataset loader，向
   `sglang.benchmark.serving.cli_main()` 提供 input_ids；调度、并发限制、
   流式 HTTP、预热、计时和指标计算都使用 SGLang benchmark。

这是固定长度性能负载，不是选择题准确率评测。ignore_eos 后的 1024 token 不做答案评分。
输入或输出计数不符的运行不进入有效结果。

## 执行

在仓库根目录，先提交/固定本地适配改动。每次对比使用同一代码和配置：

```bash
cp benchmark/lagom/overall.example.json /tmp/lagom-overall.json
# 编辑 model、draft、python；检查 env、host ratio、backend、端口。
# 环境需有本仓库 SGLang、匹配的 sglang_router、Mooncake、datasets、transformers。

.venv/bin/python benchmark/lagom/longbench_v2_fixed.py \
  --tokenizer /mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731 \
  --output /mnt/bench-data/lagom-longbench-v2 \
  --input-lens 32768 65536 131072 --max-requests 1024 --seed 42
```

离线数据加 `--dataset-path /path/to/LongBench-v2.jsonl`；联网准备可加
`--dataset-revision <HF-commit>`。目标必须是新目录；失败后检查原因，再换目录重试，
避免将部分 token bank 当作完整数据。

CPU 校验和启动计划检查：dry-run 不导入 torch、不读取权重、不启动 GPU 服务，
也不要求 token bank 已存在：

```bash
python -m unittest discover -s benchmark/lagom -p test_overall.py

python benchmark/lagom/run_overall.py \
  --config /tmp/lagom-overall.json \
  --data-dir /mnt/bench-data/lagom-longbench-v2 \
  --output /mnt/bench-results/overall \
  --dry-run > /tmp/lagom-overall-plan.json
```

执行环境补齐 PD 后，先跑 32k/C=8 三组小矩阵（每组 32 个正式请求，仍输出 1024）：

```bash
python benchmark/lagom/run_overall.py \
  --config /tmp/lagom-overall.json \
  --data-dir /mnt/bench-data/lagom-longbench-v2 \
  --output /mnt/bench-results/overall-smoke \
  --input-lens 32768 --concurrencies 8
```

当前含 PD guard 的 checkout 可追加 `--methods hisparse dhisparse`，
先验证两个 baseline；这不构成完整三组实验。

完整矩阵和续跑：

```bash
python benchmark/lagom/run_overall.py \
  --config /tmp/lagom-overall.json \
  --data-dir /mnt/bench-data/lagom-longbench-v2 \
  --output /mnt/bench-results/overall

python benchmark/lagom/run_overall.py \
  --config /tmp/lagom-overall.json \
  --data-dir /mnt/bench-data/lagom-longbench-v2 \
  --output /mnt/bench-results/overall --resume
```

`--methods` / `--input-lens` / `--concurrencies` 可选子集；组顺序始终固定。
resume 只跳过成功、且协议/代码/数据相同的格；失败项创建新 attempt，保留旧日志。
代码、token bank 或配置变化须换结果目录。输出目录有排他锁；不同目录同时运行
仍争用 GPU，应单实例执行。

`prefill_extra_args` / `decode_extra_args` / `router_extra_args` 为 JSON argv 数组，
用于本机网络和 backend 适配，所有方法共用；不得覆盖方法、buffer、窗口、并发、graph 等受控项。
默认 Mooncake 采用 `INTRA_NODE_NVLINK` + `MC_INTRANODE_NVLINK=true`；
需核对本地 transport 与 HiSparse host 接收内存兼容性。
若改为本机 RDMA 等配置，应统一三组并记录，不能退化为 fake transfer。

## DHisparse oneshot verify 核对

当前分支的 `python/sglang/srt/managers/hisparse_coordinator.py`：

- `swap_in_selected_pages_spec` 调用 `_run_swap_in_verify_window`。
- 后者将 top-k 组织为 `[num_reqs,verify_width,top_k]`，
  一次调用 `load_cache_to_device_buffer_spec_mla`。
- 因此静态 DSpark verify 是整窗口 gather/commit，不是每个 draft token 单独调 Python swap-in。

脚本通过 AST 检查两个调用并记录位置。启动后读取 P/D `/server_info`，
核对算法、窗口、TP、buffer、预取、MoE backend 和 decode graph 开关。
这是源码/配置核对，不能替代 GPU trace 对实际 dispatch、capture/replay 的验证。
不添加新的 oneshot 开关，也不将 DHisparse 改为 Lagom planner。

## 指标和产物

```text
<output>/<method>/<input_len>/<concurrency>/attempt-0001/
  launch.json
  prefill.log
  decode.log
  router.log
  benchmark.log
  prefill_server_info.json
  decode_server_info.json
  benchmark.jsonl
  status.json
```

顶层 `manifest.json` 保存协议、配置、data hash、Git HEAD/dirty diff hash、
脚本 hash、GPU 型号和 source preflight；`summary.csv` 更新各格最新 attempt。
验证 N 个请求全部成功、每条精确输入/1024 输出、总 token 数正确、无请求错误、
指标为有限正数。超时部分输出仅用于排错。

- **output TPS** = SGLang `output_throughput` =
  正式请求完成的输出 token 总数 / 正式 benchmark wall time。
  不用 input+output total throughput 或 retokenized throughput。
  包含 PD prefill、传输、排队、decode、router/client 开销、首尾填充/排空；
  不含启动、graph capture、预热、结束后指标整理。不是纯 GPU decode TPS。
- **TPOT** = 每条成功请求 `(E2E latency - TTFT) / (output_tokens - 1)`，
  沿用 SGLang 定义；主指标 `mean_tpot_ms`，保留 median/p99。
  推测解码一个流式 chunk 可包含多个 token，TPOT 不等于单 token GPU 执行时间。
- `benchmark_seconds` 为正式测量时间；`cell_seconds` 包含启动、预热、清理。
- 只对 `status=ok` 作图和比较，相同 `(input_len,C)` 下：
  TPS speedup=Lagom TPS / baseline TPS；
  TPOT speedup=baseline TPOT / Lagom TPOT。失败格保留 failed/timeout/缺失标记。

4C 请求约为四轮负载，不是无限稳态；报告保留此口径。
高 C 时留意 benchmark client/router CPU 饱和。若需定位 D 节点预取收益，
另跑已有 trace 实验，不混入本次 overall。

## 执行环境 agent 的适配清单

1. 确认可见 GPU 显存、主机 RAM/NUMA、Mooncake/router 版本，固定依赖和 model revision。
   核对 target/draft 路径、Marlin 和 attention backend。
2. **补齐 Lagom PD 运行时支持。** P 保持 DSpark context KV 注入，D 开 Lagom。
   核对 target/draft pool 传输布局、HiSparse host KV 接收、
   prealloc/transfer → decode 的 speculative state、槽位生命周期。
   验证 retract/rebootstrap、请求结束及槽位复用，不让旧预取污染新请求。
3. 验证 DHisparse oneshot dispatch、Lagom W/P 边界和实际预取命中；
   检查 P/D 预测窗口不同的 draft KV 传输，不能只删 PD guard。
4. 三组检查 CUDA Graph capture/replay，包括 batch shape 变化、
   prefetch stream/event 依赖、padded request 行。不能关闭 graph 绕开错误。
5. 运行 CPU tests、dry-run、32k/C=8 smoke，再检查 128k/C=256 资源可行性。
   buffer/host ratio 调整必须三组一致；固定最终配置及代码后开始正式实验。
6. 核对原始 JSONL 的成功数、1024 输出、TPS/TPOT，再跑 54 格矩阵。
   每格 6h 内失败则记录原因，不缩减 N 后冒充同一实验。
