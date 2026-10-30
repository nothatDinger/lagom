# V4 + DSPARK：扩展验证窗口的离线 probe 实验

本实验判断：**draft 每步一次生成 G 个新 token；其中后缀的 P 个 probe，在每个 CSA 层选出的 KV 集合，对下一步 W 个 draft token 所需 KV 集合的覆盖率是多少？** G/W 均不包含已有 anchor。逐层 Top-K 集合覆盖率是主指标，miss/搬运相关指标仅作辅助。

分支：`exp/oasiskv-trace`，基于 `oasiskv` 的 `ef8bbeae7`。这里只增加 opt-in 采集入口，不启用在线多 token 预取，不改变原有 OasisKV 的 root-only 提交语义。

默认模型与硬件：

| 项目 | 配置 |
| --- | --- |
| Target | `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/` |
| DSPARK | `/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| GPU | 4 × H100，TP=4，纯 TP，单机 |
| 数据 | `/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json` |
| 输入 | 16384 / 32768 / 65536 / 131072 token，沿用 ShareGPT 拼接规则 |
| 验证范围 | B=1，greedy，W 固定为 6 或 7 个 draft 候选，不含 anchor；target 实际输入 7/8 行 |
| 总宽度 G | 一次生成 12 个新 draft token；服务端 `speculative_num_draft_tokens=13`，额外含 anchor |
| Probe | W=6：P=0～6；W=7：P=0～5；始终满足 W+P≤G |
| DSPARK proposal | `checkpoint_gamma` 单独记录；运行时 `proposal_gamma=G=12` 个候选；所有 P 复用同一份 proposal |
| 输出 / 采集窗口 | 每请求 256 输出 token；只记录前 32 个验证窗口 |
| KV | 原生 V4 + HiSparse；Top-K 读取 target 的 `index_topk`；W=6/7 联合扫描统一常驻 `(7+1)×512=4096` |
| 路由专家 | 显式 `--moe-runner-backend flashinfer_mxfp4`；H100 使用 CUTLASS W4A16，保持 checkpoint MXFP4 存储 |
| 全局 C4 池 | `host_to_device_ratio=1`，容纳完整 prefill；不改变每请求 decode 的 `resident_units` |

**协议变更：旧版本的 G/W 包含 anchor，新版本改为新生成 token 数，并记录 `token_counting="generated"`。** 旧 `G=12,W=6/7` 实际生成 11 个、验证 5/6 个候选，不能当成新配置的结果；必须重新采集才能测试本轮 12 个 proposal、6/7 个验证候选。旧完整 trace 可重新分析，但会按实际候选数标注。新矩阵的 GPU 运行仍待验证；本地回归不代替模型实测。

独立服务仍可能产生不同 proposal/Top-K，因此质量审计使用同服务、同 proposal、同快照的基线。需要使用已能运行该仓库 V4/DSPARK 的 CUDA、FlashMLA、DeepGEMM 等环境；脚本不自动安装或下载模型依赖。

## G、W、P 的口径

这里 G 和 W **都不包含 anchor**。W=6 验证 6 个候选、最多提交 7 个 token；W=7 验证 7 个候选、最多提交 8 个 token（含 target 的 bonus）。
每步仅一次调用 draft proposer，生成 `d₁…d₁₂`；真实验证使用 `[anchor,d₁…d_W]`，P 预测使用紧随其后的 `d_(W+1)…d_(W+P)`。
P 是额外 probe，不是 W 的一部分。G=12 时，W=6/P=6 与 W=7/P=5 的 target shadow 都计算 13 行；P 较小时只取同一份 proposal 的相应前缀，不重新生成。
默认同时传入 `--speculative-dspark-block-size 12 --speculative-num-draft-tokens 13`，
为 DSPARK 一次生成 12 个候选并预留相应事务 scratch；两个参数必须匹配，单独设置后者会被 CLI 的 checkpoint 宽度校验拒绝。
trace 的 compact verify 把真实窗口限制为 W 行，额外候选不会被接受。
`--total-width 12` 固定 proposal 宽度；即使只测试 `--probes 0` 或 `--probes 1 2 3 4`，仍一次生成 12 个，target 总容量仍是 13 行。只有最大 P 满足 W+P=G；其余 P 满足 W+P<G。
多次执行的是 target 的离线基线/shadow/恢复验证，不是分别调用 draft 生成 W 和 P。每条新 trace 保存含 anchor 的完整 `proposal_input_ids`，audit 检查所有 P 与基线复用同一 proposal 及其正确前缀。

**Compressor 状态环随服务最大推测宽度分配。** 旧 C4 ring=16 的安全上限为 `16−8+2=10` 行，不能支持 G=12。
现在 G=12 选择 ring=32（安全上限 26），同时用于内存预算、attention/indexer 的原始压缩状态池和读写计划；
ring 必须整除 SWA page=256，因此不直接使用最小的 18。C128 ring 仍为 256，G≤10 的 C4 ring 仍为 16。
构造池后固定 stride，避免后续缩短 verify 宽度改变寻址。启动容量检查保留，超过页内可支持的宽度仍报错。
Compressor ring 与压缩 KV 常驻表是两种容量。本轮真实 target 窗口多了 anchor，常驻安全下界按 `(W+1)×K` 计算；联合扫描为 4096。
日志应出现 `max_draft_tokens=13, c4_ring_size=32, c128_ring_size=256`；
`runtime.json` 的 `hisparse_runtime.compress_state_ring_sizes` 应为 `{"4": 32, "128": 256}`。

加载模型前可先验证配置到真实状态池的 CPU 链路，再在 GPU 上验证原生 planner 的十二行写入与回滚历史保护：

```bash
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 python -m pytest -q \
  test/registered/unit/spec/test_oasiskv_trace_pool_capacity.py \
  test/registered/unit/mem_cache/test_dsv4_compress_write_pad.py
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py
```

对于原生 `checkpoint_gamma=5` 的 checkpoint，这属于**扩大运行时 proposal 长度的离线实验**，没有修改或重训 checkpoint。
draft 的 forward/Markov sampler 支持该宽度；离线固定窗口不调用按训练宽度校准的 confidence planner。
更宽的 draft forward 也可能改变前 5 个候选，本轮 proposal 或接受率不等同于 checkpoint 原生 5 候选。
每步只生成一次宽 proposal，不同 P 共用它的前缀；不要混合不同 G 的运行。W 的改变也会改变接受轨迹，每个 W 单独做同快照审计，不要求跨 W 窗口逐字段相同。
`manifest.json`、`runtime.json` 和 trace 记录 `total_width`、`checkpoint_gamma`、`proposal_gamma`、`probe_workspace`；分析 CSV 显式列出 `verify_width` 和 `total_width`。

**W=6/7 联合扫描统一使用 4096 常驻槽。** W=7 的真实窗口含 anchor 共 8 行，安全下界为 8×512；统一容量便于对比 W。
单独 W=6 默认使用 3584，可显式设置 4096 与扫描矩阵对齐。此前 3072/3584 的结果不能与本轮 4096 的结果视为同配置。16K 历史可能完全驻留，这不影响主指标：Top-K 集合覆盖率不要求产生 miss。
exact shadow 的每行完成原生补载后，把该行选中的历史 KV 复制到临时 C4 位置，attention 读取这些副本，
防止后面行换入时覆盖前面行仍要读取的历史 KV。当前窗口新生成的 KV 继续由事务 scratch 处理。
临时位置来自全局 C4 池的空闲页，在 prefill 完成后、每次 exact shadow 开始时借用，forward 完成后归还并恢复 free-list 顺序。
每个 TP rank 最大借用 13×512=6656 个位置，在全部 C4 层对应独立数据；不会加入常驻表，也不保留到下一真实窗口。
全局池不够时会明确报错；应增大全局池/`--context-length`，不要扩大 `--resident-units` 来掩盖缺口。

resident shadow 只给前 W 行补载，probe 历史 miss 不补载、不使用副本。
新增 `probe_workspace_d2d_bytes` 单独统计 exact 全部行（含真实前缀）的临时复制量，不计入 H2D miss、候选命中或真实常驻容量。
这仅用于离线质量采集，不代表线上采用双缓冲，也不能据其耗时推导性能。

## 执行

在 GPU 机器的仓库根目录执行，使用原来运行 SGLang 的 Python 环境：

```bash
git fetch origin
git switch --track origin/exp/oasiskv-trace
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

# 只读 checkpoint 配置并打印命令；不加载模型、不运行 GPU。
python benchmark/oasiskv/trace/run.py --mode paired --total-width 12 \
  --verify-widths 6 7 --output /srv/oasiskv/trace-plan

# 首先验证模型入口、快照恢复和接受边界。4K可以没有miss，不作为覆盖率曲线。
python benchmark/oasiskv/trace/run.py --mode paired \
  --topk-order logical-all \
  --total-width 12 --verify-widths 6 7 --resident-units 4096 \
  --output /srv/oasiskv/trace-g12-w6-w7-smoke \
  --input-lengths 4096 --output-length 32 \
  --requests 1 --max-windows 3 \
  --diagnostics --request-timeout 1800 --execute

# 正式矩阵：4 个输入长度 × 2 个 W；各 W 的 P 自动取 0..G-W。
python benchmark/oasiskv/trace/run.py --mode paired \
  --topk-order logical-all \
  --total-width 12 --verify-widths 6 7 --resident-units 4096 \
  --input-lengths 16384 32768 65536 131072 \
  --output /srv/oasiskv/trace-g12-w6-w7-16k-128k --execute
```

输出根目录的 `sweep.json` 列出各 W 的解析参数和完成状态；`w6/`、`w7/` 各有独立 manifest、paired trace、audit 和 analysis。
`--mode paired --verify-widths 6 7` 顺序启动两个服务，每个 W 一个，各自串行处理全部四档输入；P 扫描在同一 proposal/快照内完成，不为每个 P 重启服务。
四档输入由同一最长 ShareGPT prompt 取前缀，W 两组使用相同 seed/数据和输入长度集合；`prompts_sha256` 可核对输入一致。

128K 输入默认把 `--context-length` 与 `--max-total-tokens` 设为 **139264**，给输出、C4 scratch 和 prefill 页留余量；
16K/32K 单独运行仍至少用 40960，64K 单独运行默认 73728。显式覆盖 context 时仍会检查容量，不能沿用 40960 跑 64K/128K。
全局 C4 池仍保持 `host_to_device_ratio=1`，每请求 decode 常驻仍为 4096，不随输入长度增长。
`--snapshot-gib` 仍是 32 GiB/rank 的上限，不是预留量；长上下文的实际占用查看 `snapshot-inventory.json`，超限会报出所需大小，不会自动放宽主机内存预算。

已有本地实验分支时使用 `git switch exp/oasiskv-trace`。输出目录必须不存在，以防混合旧 trace。默认使用 GPU `0,1,2,3`；可通过 `--gpus` 修改。推荐 `--mode paired`：只启动一个 probes 服务，基线来自该服务每个窗口的同快照对照。`--mode experiment` 仍顺序启动独立的 control/probes/full-control 服务，各自重新加载模型，共用 `prompts.npy`；它增加跨服务复现性和checkpoint 原生接受长度参考。请求串行执行。
`--port` 为基准端口：experiment 的三组分别使用 base、base+1、base+2；单独运行一个模式使用 base。
启动前用 `SO_REUSEADDR` 的 bind/listen 检查端口，与服务端语义一致，避免把旧连接的 TIME_WAIT 误报为活跃 listener。
已有活跃 listener 仍报错；预检查不持有服务 socket，不能防止其他作业恰在启动之间抢占端口，也不会清理其他作业。
Linux runner 在启动服务前启用 child subreaper，接管服务遗留的孤儿进程，退出时清理本次服务的所有后代，包括调用 `setsid()` 脱离原进程组的 worker；已有无关服务不会被清理。
清理不依赖 wrapper 的 `poll()`，TERM 超时后升级为 KILL，仍存活则写 `teardown-failure.json` 并中止后续组别。
GPU工作区若已有针对 `eb91f01a4` 的未提交修补，请先保存diff，并在新的worktree检出更新后的分支验证，
避免把旧的初始化补丁重复覆盖到新代码上。运行manifest同时记录HEAD和`source.diff`的SHA256。

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
只有 probes 产生**请求ID为 `trace-i4096-r0` 的**有效窗口及配对基线、同进程 `audit.json` 为 `passed=true`，且 `analysis/report.json` 有匹配样本，
才进入 16K/32K 正式采集。这个修复没有在本地 GPU 上验证，不据启动顺序回归宣称已有实验结果。

## 首个真实请求挂起：修复与定位

旧代码存在可在CPU复现的接纳死循环：V4架构把默认 `swa_full_tokens_ratio` 设为0.1，
所以 `max_total_tokens=40960` 时SWA池只有4096槽。4096-token chunk加256-token页预留需要4352槽。
本实验 `ignore_eos=true` 且关闭radix cache，走 `PrefillAdder.add_one_req_ignore_eos()`；
旧分支直接返回 `NO_TOKEN`，没有普通分支已有的“整个SWA池也装不下时缩小chunk”的处理。
等待、减少running请求数都不能增加池的总容量，因此256-token warmup能完成，4K请求却一直排队。

修复复用普通接纳路径的SWA chunk上限：4K、输出32的回归按3584＋512处理；
临时内存压力仍等待，不挤占运行中请求的headroom。不改变输出长度、spec接受规则或trace隔离语义。
这个根因已经过CPU失败→通过回归，但仍需GPU复测确认现场没有第二个阻塞点。
普通 `Prefill batch` 日志在结果处理时打印，单凭没有该日志不能区分“未接纳”和“forward未返回”。

现在可分别执行无trace和有trace的4K诊断，保持旧实验的全局C4比例2，以隔离接纳修复。
每条命令独占4卡、单独加载模型；输出目录必须是新的，不要同时运行：

```bash
# 不设置 SGLANG_OASISKV_TRACE_CONFIG；脚本也会清除继承的同名变量。
python benchmark/oasiskv/trace/run.py --mode untraced --diagnostics \
  --input-lengths 4096 --output-length 32 --requests 1 --max-windows 3 \
  --host-to-device-ratio 2 --request-timeout 600 --watchdog-timeout 300 \
  --output /srv/oasiskv/trace-4k-untraced-fixed --execute

# 固定V的control，有观测但不做shadow快照。参数与上面一致。
python benchmark/oasiskv/trace/run.py --mode control --diagnostics \
  --input-lengths 4096 --output-length 32 --requests 1 --max-windows 3 \
  --host-to-device-ratio 2 --request-timeout 600 --watchdog-timeout 300 \
  --output /srv/oasiskv/trace-4k-control-fixed --execute
```

上述 untraced/control 模式只采集，不自动出audit/coverage报告。无trace臂沿用普通DSPARK planner；
control臂固定V，它们用于判断首次prefill/trace入口的阻塞，不用于证明完整输出轨迹相同。
两臂都通过后再跑上面的默认experiment smoke（含probes），最后跑16K/32K。
`control` 本来就不执行shadow的 `torch.cuda.synchronize()`/快照分支；它失败不能直接归罪于shadow重放。

`--diagnostics` 给 `server.log` 增加CPU进度标记，各rank保留PID和request ID：

| 最后可见阶段 | 下一步检查 |
| --- | --- |
| `queued` / `admission`返回`NO_TOKEN` | `full_free`、`swa_free`、`swa_capacity`、`swa_needed`；拒绝日志每30秒最多一次 |
| `prepare-extend-enter`，无`returned` | 接纳后的批次准备和KV分配 |
| `target-prefill-enter`，无`returned` | target首个真实prefill、CUDA/通信/内核编译；看watchdog栈 |
| `draft-kv-inject-enter`，无`returned` | DSPARK接收target hidden并写draft KV的路径 |
| `draft-propose-enter`，无`returned` | DSPARK生成本次proposal的路径 |
| `trace-enter` / `trace-forward-enter` | decode验证/观测路径，而非首次prefill接纳 |
| `snapshot-sync-enter` / `snapshot-copy-enter` | probes组的快照阶段 |

这些标记不增加CUDA同步；`returned`只表示Python调用返回，不宣称GPU任务已完成。
`server_info.json` 保存ready后的完整解析配置，`runtime.json` 另记录full/SWA/C4池容量。
`request-progress.json` 保存最近请求的提交/完成状态；异常写 `failure.json`，保留server日志。
client请求超时和scheduler watchdog分别设置，默认后者600秒；长时间shadow可显式调大。

原脚本还有一个32K容量缺口：`host_to_device_ratio=2`、full池40960时，全局C4物理池仅约5120单元，
且DSPARK scratch还占用其中部分。32K prompt需要8192个C4单元，当前HiSparse在最后一个prefill chunk后才卸载历史，
所以减小chunk不能解决这一容量问题。脚本现在默认比例1，并在计划阶段拒绝明显装不下的组合。
这只扩大prefill可用的全局物理池，**每请求decode驻留仍按 `resident_units` 裁剪**；
旧口径 W=6、K=512 时容量为3072；本轮新口径 W=6/7 扫描为4096。16K初始历史可全部常驻，32K以上仍会超过容量；主指标不以miss为分母。

MXFP4方面，脚本显式选 `flashinfer_mxfp4` 并设置 `SGLANG_DSV4_FP4_DEQUANT=0`。
CUDA上 packed MXFP4 不再静默落到普通FP8 MoE；不支持的runner（包括原来的auto回退）会提前报出可用选择。
共享专家的FP8路径保留。手工使用 `SGLANG_DSV4_FP4_DEQUANT=1`＋auto仍可走已有解包路径，
但会扩大专家权重显存，不属于本实验默认配置。

GPU 测试方在 h100-8（10.72.1.175）使用 CUDA 13 wheel 工具链时，还需要以下环境设置；这些是该机器的配置，不是所有 CUDA 环境的通用要求：

```bash
# CUDA_HOME 使用现场已验证的 .venv/.../nvidia/cu13 绝对路径。
# 先确认该目录内 lib64 -> lib 及链接器需要的未带版本 soname 已准备好。
export NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
```

runner 继承这些设置，并在 `launch.json` 记录上述变量及 `CUDA_HOME`。
关闭 DeepGEMM 启动时的大网格预编译不等于关闭运行时 JIT；缓存目录继续使用现场已验证的持久化配置。

## Stage 3 恢复校验失败：Top-K 顺序与集合

`ae1545f24` 用 Python 列表精确比较 trial/replay 的整个逐层观测。
V4 HiSparse 的原生 `topk_v1.cuh::radix_topk()` 用 `atomicAdd` 分配输出位置，**不保证同一组选中 ID 的排列顺序**。
因此仅排列变化就能导致所有层报不同，不能据此认定快照恢复或 observation override 泄漏。
每次 `observe()` 都创建独立对象，Top-K 已转成 CPU list；变宽重放、张量覆盖及异常退出有 CPU 回归覆盖。

现在窗口内和独立进程 audit 都逐 query 比较 ID **多重集合**，逐 miss event 比较实际搬运 ID 多重集合：保留行边界、重复次数、padding 与驻留集合校验。
原始 Top-K / miss 顺序继续保存在 trace；不修改传入 attention 的 GPU 张量。
若阈值并列导致选中 ID 本身改变，或 miss/驻留/argmax token 改变，仍拒绝；本修复不把这些情况视为噪声。

`replay-audit.jsonl` 记录每个窗口是否通过、`topk_order_only_layers`、差异字段、argmax 和有限 logits 的最大绝对差。
**相同 argmax 不代表 logits 完全相同，也不能单独证明状态恢复完整。** 如果仍拒绝，查看 `rejection.json` 与 `rejected-trace.jsonl.gz`，区分 `topk`、`miss_events`、`resident_after` 的真正变化。
拒绝记录使用 `rejected_window` / `rejected_probe`，不进入正常 trace；audit/analyze 拒绝带 `rejection.json` 的目录，即使其中已有前几个窗口的有效记录。

P 行是 P 个 query 的 Top-K，和搬运次数没有直接等价关系。4K 输入约有 1024 C4 单元，容量 3072 时完全可能全部驻留；此时 exact 和 resident 的 `probe_demand_h2d_bytes=0` 正常。
先用上方 4K experiment smoke 验证采集与隔离，再跑 16K/32K 研究容量压力；不能用全驻留的 smoke 推导预取命中率收益。

### 后续证据：首层换序、深层集合漂移

一种需要验证的因果链是：首层选中 ID 相同但排列不同 → attention 浮点归约顺序改变 → hidden state 微小差异 → 后续 indexer 分数/Top-K 改变。
此时后续 indexer 接收到的张量未必位相同，不能直接归因于“相同 indexer 输入产生不同分数”。
`1024` 选 `512` 本身也不能证明边界存在大量同分候选，需要实际分数证据。
全驻留时不同 Top-K 都可能零 miss 且不改变驻留集合，这两个相同字段对排除恢复问题的约束有限。
最终三行 argmax 相同不是“21 层的 logits 位相同”；仍须保留隔离检查。

新增两个显式诊断选项：

- `--mode replay-check`：相同快照 → 基线 forward → 恢复 → 基线 forward，不运行 P 个 probe 或任何变宽 shadow；仍执行真实模型与全部严格校验。只产出诊断，不生成覆盖率报告。
- `--topk-order logical`：仅在 trace 的 target verify 路径，原生 Top-K 选择后、HiSparse 换入及 scratch 映射前，按逻辑 C4 ID 排序，同时对原始 ID 和 page 索引施加同一排列，padding 放末尾。随后 attention 使用固定的逻辑顺序。此选项不更改当前集合，不规定同分候选的取舍，也不保证其余 kernel 确定性。
- `--topk-order logical-all`：覆盖 target 的 prefill、verify 和 decode C4 路径，保持各路径原有的 v1/v2 Top-K 选择算法，仅对选出结果排序。prefill 额外取得 v2 已支持的 raw ID 输出，不因为申请 raw ID 就切换到 v1。

默认仍为 `--topk-order native`。`logical` 和 `logical-all` 是分别标记的实验变体，会影响浮点计算顺序，不能保证与 native 逐 token 相同。
控制/探针组必须使用同一种设置；`runtime.json`、每条 trace、`analysis/report.json` 都标记 `topk_order`，audit/analyze 拒绝混用。
untraced 不安装该 hook，因此只允许 native。`logical` 不覆盖 prefill；`logical-all` 覆盖，但不承诺消除全部 CUDA/TP/MoE 数值不确定性。

在已有 CUDA/JIT 环境设置下，可先各跑一个不含 probe 的双重放诊断（每条命令重新加载一次模型）：

```bash
python benchmark/oasiskv/trace/run.py --mode replay-check --topk-order native \
  --gpus 1,5,6,7 --input-lengths 4096 --output-length 16 \
  --requests 1 --max-windows 2 --diagnostics --request-timeout 1800 \
  --output /srv/oasiskv/replay-native-new --execute

python benchmark/oasiskv/trace/run.py --mode replay-check --topk-order logical \
  --gpus 1,5,6,7 --input-lengths 4096 --output-length 16 \
  --requests 1 --max-windows 2 --diagnostics --request-timeout 1800 \
  --output /srv/oasiskv/replay-logical-new --execute
```

如果 native 失败而 logical 通过，支持“KV 访问顺序引入数值漂移”的解释；仍不是恢复完整性的独立证明。
接着执行包含 probes 的完整 smoke：

```bash
python benchmark/oasiskv/trace/run.py --mode experiment --topk-order logical \
  --gpus 1,5,6,7 --input-lengths 4096 --output-length 32 \
  --requests 1 --max-windows 3 --skip-full-control --diagnostics \
  --request-timeout 1800 --output /srv/oasiskv/probes-logical-new --execute
```

这条 `logical` 命令保留用于复现 fix3/fix4 的 verify-only 诊断。当前执行方案见下方 paired 实验。
若固定顺序后仍有集合差异，应继续采集首次分歧层的 indexer 输入、分数和 K/K+1 边界，区分同分选择、分数漂移与状态恢复问题，不能为了生成曲线忽略集合变化。
`rejection_layer_diffs` 保留逐层 `resident_after_equal` / `miss_events_equal`，`compare_topk()` 进一步区分逐行换序、换 ID、重复次数及缺行。

### 独立 control 仍不同：改用同进程、同快照的配对实验

`audit.py` 只比较 `kind=window` 的 **真实基线 verify** Top-K，不比较 `kind=probe` 的反事实 Top-K。
被拒绝的候选虽不进入输出，仍参与本轮 verify、选择 KV 并可能改变驻留状态；因此相同的 committed output 不能证明 KV 基线相同。
报告中的 `resident_after` 也仅 42/63 相同，删掉 `input_ids` 和 `topk` 后仍有 21 条差异。
跨服务不一致不证明 shadow 污染；窗口内恢复检查与跨服务基线检查分别回答不同问题。
`fd58f279f` 的 control/control 证据进一步确认：本环境不满足跨进程逐字段复现的前提；不再要求先解决全模型确定性，才能研究一条实际轨迹上的预测质量。

V4 DSPARK 的 `DSparkAttention` 使用 `compress_ratio=0` 的 **SWA** 路径，没有 C4 indexer。
但 target prefill 的 hidden state 通过 `inject_target_hidden()` 写入 draft KV，旧 `logical` 未固定这段 target C4 访问顺序。
`logical-all` 已覆盖这段路径，但现场证明它仍不足以实现跨服务复现。相同 token IDs 不代表 hidden/KV 内容位相同；而 indexer 先根据 query、indexer KV 打分选 Top-K，再执行 HiSparse 映射/换入，**驻留集合不是 Top-K 的直接筛选条件**。不能仅凭当前记录认定“拒绝 draft 导致驻留不同”就是所有分歧的根因。

新的主校验固定 **同一次 draft proposal 和 pre-forward 快照**：

1. 执行 V 基线，保存逐层观测、logits 和交给后续 draft 的 hidden state。
2. 每次恢复该快照，运行各 P 的 exact/resident shadow；不提交它们的输出。
3. 再恢复快照并重跑 V。要求逐 query Top-K 多重集合、miss、驻留集合相同，且 logits、hidden state 精确相等。
4. pre-shadow predictions 在 CPU 独立计算固定 V 的 greedy 接受结果，与实际 accept kernel 的 commit 长度和输出 token 核对。
5. 保存 `paired-baseline.jsonl.gz`，离线 audit 检查与 final windows 的逐字段配对、snapshot/proposal 身份、完整 P/mode/layer 记录和请求输出轨迹。

没有把 Top-K 降级为无需验证的观察字段；改变的是对照的来源。这个检查约束所观测的状态及输出，不宣称证明了任意未来轨迹或所有内存逐字节相同。

推荐先执行一台服务的 smoke；它不再重新加载两个独立 control：

```bash
python benchmark/oasiskv/trace/run.py --mode paired --topk-order logical-all \
  --gpus 1,5,6,7 --port 31010 --input-lengths 4096 --output-length 32 \
  --requests 1 --max-windows 3 --diagnostics --request-timeout 1800 \
  --output /srv/oasiskv/paired-4k-new --execute

# 上述同进程 audit 通过后，先少量采集 16K/32K。
python benchmark/oasiskv/trace/run.py --mode paired --topk-order logical-all \
  --gpus 1,5,6,7 --port 31010 --input-lengths 16384 32768 --output-length 64 \
  --requests 1 --max-windows 8 --diagnostics --request-timeout 3600 \
  --output /srv/oasiskv/paired-long-new --execute
```

`paired` 自动执行 `audit.py probes/ --scope within-server` 和 analysis。`--mode probes` 仍是仅采集入口。
同进程失败写 `audit-failure.json` 并停止分析；旧 trace 缺少新配对基线/hidden 证据时必须重新采集，不能直接改标为通过。
`--mode experiment` 同样强制同进程 audit，另生成 `cross-server-audit.json` 或 `cross-server-audit-failure.json`；跨服务差异仍标记 `passed=false`，默认不阻塞这条轨迹的离线分析。
若要恢复原先的两道硬门槛，加 `--require-cross-server-match`。直接调用两个目录的 `audit.py` 仍是严格跨服务校验，未悄悄缩减字段。
21 个 C4 层、3 个采集窗口、P=1/2/3、exact/resident 两种模式，预期 **63 条 window + 378 条 probe**；不足时检查实际生成窗口数，而非把不同组的 trace 混合。
这时还有 63 条 pre-shadow paired baseline，单独存储，不混入分析的真实窗口。

`history_coverage` 比较 probe 预测集合与 **同一 probes 服务下一真实窗口** 的需求，属于离线预测潜力；它并非把 shadow 分支自身当作 ground truth。
因此 fix3 的 0.796→0.878→0.945 可以描述该次采集的预测潜力，不能称为 H2D 命中率。
`analysis/report.json.validation` 保存本次审计的作用范围，并用 SHA256 绑定到 trace；不带 `--audit` 手工分析则标记 `not_audited`。
单条配对轨迹的 P 对比不需要跨服务完全确定性，但要声称平均预测质量，仍应在不同请求/seed 的独立 paired 运行上检查波动；本脚本不自动量化跨服务方差。
4K 全驻留时无候选/无 miss 的比例应为 null。16K/32K 增加容量压力，但历史超过容量仍不保证每个窗口都有 miss；要检查实际 miss 字节分母。

真实 CUDA kernel 回归不需要加载 checkpoint：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/attention/test_oasiskv_trace_topk_order.py
```

该测试重复调用 V4 原生 v1/v2 Top-K，验证固定分数下的集合、短行/padding，以及排序后的 raw/page 对齐；它不代替上述完整模型双重放和 probes 验证。本地 CPU 测试中的 executor 仍为模拟实现，不应称作真实 GPU replay 已通过。

新实验配置列出允许采集的request IDs：server warmup只执行正常验证，不保存window、不做shadow，
不进入覆盖率和接受长度统计。原日志中的 `history_units=1–2` 不能代表4K请求；
4K真实历史应约为1024个C4单元，16K约4096、32K约8192。
4K通常全驻留，miss分母为0是合理的；逐层集合覆盖率仍可计算。正式矩阵按16K/32K/64K/128K分别报告。

当前 `--total-width G` 默认 12，不含 anchor，独立于 checkpoint 原生候选数；运行时 proposal gamma 为 G，target 总行数为 G+1。
未显式传 `--probes` 时自动扫描 0..G−W，所有 P 共用固定 proposal 宽度。
旧的 `--verify-candidates V` 参数仍可用，等价于 `--verify-width V`（此处 V=W，均为候选数）。control 与 probes 使用相同运行时 proposal gamma 和 W。
untraced/full-control 使用 checkpoint 原生候选数；前者不做采集，后者用于提供原生接受长度参考，不属于本轮 paired 矩阵。

Top-K 使用模型原生设置，不用通用 Qwen 实验的 2048 代替 V4 的 indexer 配置。
paired 默认容量是 `(W+1)×index_topk`。若 experiment 还包含比 W 更宽的原生 full-control，则共同容量也必须满足原生宽度下界；
所有组实际容量写入 `runtime.json`。只有 opt-in trace 的 exact shadow 使用上述临时副本；普通 DSPARK 的容量检查不变。

### 16K 首窗口误报 C4 4095 被淘汰

GPU 测试方在 `c1d98fda2` 报告的 `references evicted C4 4095` **不能直接证明容量不足**。
6 行各取 512 个 ID，其并集最多为 3072；3072 是最坏情况下的历史并集上界，不是“一有行间差异就不够”的临界值。
请求的真实历史规模以 trace 的 `prefix_len/history_units` 为准，不能用不同层的 Top-K 并集大小代替全部历史。

代码中有一个可在 CPU 使用真实 allocator/coordinator 复现的元数据缺口：

- V4 allocator 裁剪长 prefill 时，把最后一个 C4 的物理位置保留在 `slot=device_buffer_size`，即 LRU 之外的 reserved slot。16K 对应的最后一个 ID 正是 4095。
- coordinator 原来只登记前 3072 个 hot slot 的 token ID，reserved slot 的标签仍为 −1。换入 kernel 的 newest 快速路径返回该有效地址，trace 的元数据检查却认为 token 不在其中。
- 已补齐标签，依据 allocator 裁剪前的物理映射验证该 KV 确实被保留。映射为空的 direct-to-host 路径不会被错误标记成已驻留。
- 同一 token 可同时保留 hot/reserved 两份有效副本；guard 改为检查 `(token, physical_location)`，避免单个 `token→location` 字典覆盖另一份有效地址。真正的陈旧地址仍拒绝。

这次修复不扩大物理容量，不改变 P、V、Top-K 或输入长度。reserved slot 原先已经存在。
若仍报错，错误现在包含实际 `verify_width`、历史并集大小、hot 容量、返回地址、该 token 的已登记地址和该地址的登记 token，需据此区分真实淘汰与元数据不一致。
一个窗口首先执行不含 probe 的 V 基线，因此“首窗口还没有 trace 就失败”也不能直接断言失败发生在 P=3 的六行 forward。

可先在 GPU 上跑无需加载模型的实际 H2D kernel 回归：覆盖 4096/8192 历史、6×512 行选择、3072 hot slot，以及 newest 双副本；它检查最终所有 query 引用的 KV 字节仍正确。

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hisparse.py \
  -k verify_rows_keep_live_kv

python benchmark/oasiskv/trace/run.py --mode paired --topk-order logical-all \
  --gpus 1,5,6,7 --port 31010 --input-lengths 16384 --output-length 32 \
  --verify-width 6 --probes 1 2 3 4 5 6 --resident-units 4096 \
  --requests 1 --max-windows 3 --diagnostics --request-timeout 1800 \
  --output /srv/oasiskv/paired-16k-w6-p6 --execute
```

16K 同进程 audit 通过后，再按原方案采集 32K。CPU 回归不代替上述 GPU kernel 和完整模型验证；4K 已有结果仍只作为全驻留 smoke。

### Miss kernel 崩溃：先检查计数扫描，再检查物理地址

`9bf6867e7` 的 GPU 测试被报告在 `hisparse.cuh` 的 kernel launch 处出现 illegal instruction。
失败的无模型测试直接把物理地址设为 `0..3072`，不经过 coordinator 的 hold/release；因此仅凭这一报错不能断定存在未分配的 `dst_loc=-1`。

已复现并修复一个 CUDA 计数扫描缺陷：256 线程每轮只有 8 个 warp 计数，旧代码却用 32 个 lane 扫描并写回后续计数区。
这会把前缀和当作下一轮计数再次累加，破坏 hit/eviction 数和 shared-memory 槽位索引。生产默认的 960 线程（30 个 warp）也受影响。
hit/eviction 与 miss 两处扫描现统一限制在本轮 `NUM_WARPS` 个计数内；ROCm 原有正确边界保留。
另增加 LRU、淘汰槽、物理地址断言，无效地址会明确拒绝，不会静默漏载历史 KV。

CPU 回归编译并执行实际 CUDA/ROCm 扫描函数及调用片段，用 barrier 模拟 warp shuffle；旧代码 4 项失败，修复后 16 项通过。
这证实了扫描错误，不替代完整 GPU kernel 验证，也不单凭 CPU 结果断言现场全部崩溃都已解决。
GPU 机器先按顺序运行下面两项；第一项只更新元数据、完全不读取 host KV，可区分扫描与搬运故障：

```bash
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/kvcache/test_hisparse.py \
  -k swap_plan_at_3072_capacity_without_copy
PYTHONPATH=python python -m pytest -q test/registered/kernels/ops/kvcache/test_hisparse.py \
  -k verify_rows_keep_live_kv
```

两项均覆盖 256/960/1024 线程。第二项保留 6 行无 workspace 和 12 行带 workspace、16K/32K 历史、真实 H2D/D2D 及最终 KV 字节校验。
若 CUDA 报错，使用新进程重试；出错后的 context 不可继续用来判断后续参数组合。两项通过后再加载模型运行 paired 实验。

可用参数见 `python benchmark/oasiskv/trace/run.py --help`。`--max-windows` 是每个请求的采集上限；不会伪造接受长度来凑窗口。若生成太短、没有连续两个验证窗口，分析会拒绝给出结果。想覆盖生成中后期，应增加 `--output-length` 和 `--max-windows`，并相应增加 `--context-length`。

## 配对基线及独立参考组

`--mode paired` 仅启动下面的 probes 组，该组内部的 pre-shadow V forward 就是同快照控制基线。
`--mode experiment` 额外启动独立 control/full-control，它们提供复现性与接受长度参考，不作为下一真实窗口 ground truth 的来源。

1. **control，P=0**：固定 V 的普通 DSPARK 接受流程，无 probe，无预测预取。
2. **probes**：同一个固定 V 基线轨迹，每个采集窗口执行 P=1..G−W 的 exact、resident 影子 forward；P=0 复用同快照基线，不额外 forward。这些 probe 不进入接受流程、不提交 KV，也不执行后台预取。最后恢复并重放原来的 V 窗口，走普通 DSPARK 接受流程。
3. **full-control**：用 checkpoint 原生候选数生成并验证完整候选，无 probe。用于观察扩大 proposal 长度/限制真实 V 后接受长度的变化，不强行逐窗口配对。Smoke 可以省略这一组。

每个 shadow 仍在**一次 target forward** 中计算 `[anchor, W 个验证候选, P 个 probe]`，让 probe 的 hidden state 经过所有 target 层。离线实验为做对照而多次重放；这不代表未来线上实现必须额外启动这些 forward。

P=0 没有新的预测集合。验证行自己的 Top-K 已经为本轮 attention 获取，不能把它重新统计成下一轮新增预取。
P=0 会记录 exact/resident 两条相同的零 probe 基线，Top-K 行列表为空、额外 H2D/D2D 均为 0；不会因此多跑两次 target forward。
存在分母时，预测集合覆盖率和潜在 miss 覆盖率均为 0；分母为空时仍为 null，candidate_precision 因没有候选而为 null。
这不是“GPU 没有任何 KV”，也不是总驻留集合的覆盖率，不能与 `history_coverage` 的预测集合口径混用。

## exact / resident 的具体语义

- **exact**：所有 W+1+P 行使用原生稀疏 Top-K 和按需补载，历史 KV 逐行复制到临时位置保护其读取生命周期。它是相同 draft 分支的精确稀疏 attention 参考，不是 dense attention，也不是下一轮命中率的数学上界。
- **resident**：前 W+1 行照常补载。probe 先记录原始预测 Top-K，再仅映射其中已驻留的历史项，未驻留项的 attention 索引置为 −1；**不调用 probe 历史补载 kernel**。本窗口已计算的 C4 仍经过现有事务 scratch 映射和因果限制，SWA/C128 路径照常执行。

resident 的策略是“probe Top-K 与本层真实验证后驻留集合的交集”，不是把任意驻留项补满 K。这个选择会影响深层 hidden state，因此报告必须保留 mode 标签。记录 probe Top-K 的动作发生在这一层 attention 之前；下一层的 Top-K 已受前一层近似 attention 影响。

真正验证的接受规则沿用原来的 `accept_and_finalize`，用 compact layout 把接受边界限制为 W 个候选；最多提交 W+1 个 token。Probe logits 即使算出来也不参与基线接受判断。

## 隔离与正确性检查

离线采集使用**主机状态快照**：覆盖 target 的 SWA、C4/C128 KV、indexer KV、两类 compressor state、请求映射、HiSparse 驻留/LRU/scratch 表，以及 RNG 状态。原有 GPU 地址不变，恢复时原地写回。每次 shadow 前恢复同一个 pre-forward 状态；shadow 期间不执行 `commit_hidden` 或 C4 commit。

这不是线上预取的 staging buffer。没有增加线上读写双缓冲；快照只属于实验工具，代价不纳入性能结论。

会执行以下校验：

- 窗口内：shadow 前后的基线重放必须得到精确相等的 logits/hidden state、逐 query Top-K 多重集合、逐行实际 miss 多重集合和驻留状态。随后核对实际 accept 与 pre-shadow greedy 参考结果。
- 同进程离线审计：持久化的 paired baseline、final window、replay diagnostic 与输出记录必须完整且一致；失败停止分析。
- 独立进程复现性诊断：完整输出、接受长度、验证输入、Top-K、miss 和驻留仍全部比较。默认单独报告；显式要求 `--require-cross-server-match` 时，差异也会停止分析。

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
    server_info.json / request-progress.json
    failure.json               # 仅运行异常时存在
    teardown-failure.json      # 仅服务后代清理失败时存在
  probes/snapshot-inventory.json
  probes/paired-baseline.jsonl.gz # 同一 proposal/快照的 pre-shadow V 基线
  probes/replay-audit.jsonl     # 每窗口重放检查、Top-K 仅排列变化、logits 差异
  probes/rejection.json         # 仅恢复校验拒绝时存在；阻止该目录进入分析
  probes/rejected-trace.jsonl.gz # 被拒绝窗口/探针原始观测，独立于有效 trace
  audit.json                    # 同进程配对检查，必须 passed=true
  audit-failure.json            # 仅同进程校验拒绝时存在
  cross-server-audit*.json      # experiment 的独立进程复现性诊断
  acceptance.json               # 每种 V 的 A 分布、平均提交 token 数
  analysis/samples.csv           # 每窗口、每层、P、mode 的明细
  analysis/groups.csv            # 按输入长度、层、W、实际 A、P、mode 聚合
  analysis/layers.csv            # 每个CSA层与P的主指标汇总，不按接受长度拆开
  analysis/report.json
```

trace 的 `layer` 是 C4 层序号；`runtime.json` 给出到原模型层号的映射。C4 单元代表 4 个原始 token，搬运字节取运行时 `item_size_bytes`，不把整个 C4 page 的 padding 算成每个选中单元的传输。

配对键为 `(run_id, request_id, generation, window_id, layer, head)`；window_id 是请求内的连续编号，**不按请求池槽号或全局 forward 计数配对**。要求下一窗口 `prefix_len = 当前 prefix_len + 实际 commit_len`。提前拒绝时也按这条真实路径配对，不能按 V 推测下一位置。

**主指标 `next_step_coverage`：本步 P 对下一步 W，逐 CSA 层计算。** 对层 ℓ，设 `S(t,ℓ,P)` 为本步 P 个 probe 的 Top-K 并集，`D(t+1,ℓ,W)` 为下一真实 step 的 W 个 draft 位置的 Top-K 并集：

```text
coverage(t, ℓ, P) = |S(t, ℓ, P) ∩ D(t+1, ℓ, W)| / |D(t+1, ℓ, W)|
S = 本步 probe 记录的 P 行 Top-K 并集
D = 下一步 window.topk[1:W+1] 的并集（排除第 0 行 anchor）
```

去掉 −1 padding、去重，不减驻留、不筛 misses、不限制数量。D 使用下一步实际执行的全部 W 个候选 query，即使该步最终拒绝部分候选，也不把分母改为接受 token 的需求。P 不与本步 W 比较，也不把本步 proposal 的后缀当作下一步 ground truth。

`samples.csv` 每行是一对相邻 step、一个 CSA 层及 P/mode，包含 `window_id`、`next_window_id`、`next_snapshot_id`、`covered_next_units`、`next_all_units`、`next_step_coverage`。`layer` 保留 C4 层序号，`model_layer` 是原 target 模型层号；旧 trace 从同目录 `runtime.json` 的映射恢复，缺失时留空，不猜测层号。
`layers.csv` 按输入长度、W/G、P、mode 和每个层单独汇总，不混合不同 CSA 层：`mean_step_coverage` 是有非空需求的各 step 覆盖率平均值，`next_step_coverage` 是交集数之和除以需求数之和。`groups.csv` 额外按实际接受长度分组，用于诊断。
本采集覆盖 V4 中带 indexer Top-K 的 C4 CSA 层；C128 压缩注意力和 SWA 没有这套逐行 Top-K 选择，不混入这些层的覆盖率曲线。

主指标衡量逻辑 C4 ID 的选择集合重合。未提交分支中同位置 ID 并不能证明 KV 内容相同，也不能据此声称异步预取命中；下面的历史/搬运辅助指标会另行过滤尚未提交的 KV。

**辅助预取候选仍使用 P 行 Top-K 的完整并集，不设置固定上限。** 令 `H` 为源窗口已提交的历史，`R` 为本层真实补载后的驻留集合，则历史预测集合为 `U_P = S(t,ℓ,P) ∩ H`，候选预取集合为 `F_P = U_P − R`。P=0 时二者为空。

每个 P 的候选数取实际 `|F_P|`，不做频次投票筛选或截断，也不强行补到 P×K。K=512 时，P=0～6 的原始条目数上界依次为 0、512、1024、1536、2048、2560、3072；去重、排除未提交历史及扣除驻留后，实际数量通常更小。各 P 的 shadow 使用各自观测到的 Top-K，不人为保证曲线单调。

旧版 `--prefetch-units` 默认 256，只截断离线分析中的候选集合，不控制模型或 GPU 搬运。它影响 `potential_unique_miss_coverage`、候选量及 precision；旧版 `history_coverage` 已使用完整预测并集，未受该上限影响。本轮取消这种固定预算口径：两个脚本均可省略该参数；兼容写法为 `run.py --prefetch-units all` / `analyze.py --max-prefetch-units all`，传入旧的数字值会在执行前报错。

`report.json` 标记 `analysis_version=3`、`prefetch_policy="full-probe-union"`、`prefetch_cap_units=null`；所有 CSV 也包含 `prefetch_policy`，不要与旧的固定预算报告混合。该变化只涉及离线集合统计，W/P、index_topk、常驻容量及真实接受路径不由候选预算改变。

核心字段：

| 字段 | 含义 |
| --- | --- |
| `next_step_coverage` | **主指标**：本步 P 的完整 Top-K 并集，对下一步 W（不含 anchor）需求并集的覆盖率 |
| `probe_topk_units` / `covered_next_units` / `next_all_units` | 主指标预测集合大小 / 与下一步需求的交集大小 / 下一步需求集合大小 |
| `history_coverage` | 对下一窗口需要的、源窗口已有历史 KV 的覆盖率 |
| `prefetch_candidate_units` / `prefetch_candidate_bytes` | P 行预测的全部历史候选去重、扣除驻留后的数量 / 潜在搬运量，无截断 |
| `next_miss_bytes` | 下一真实窗口 kernel miss plan 的实际 H2D 字节；重复补载按实际次数计 |
| `potential_unique_miss_coverage` | 候选集合覆盖下一窗口不同 miss 单元的比例 |
| `potential_first_fetch_bytes` | 潜在可提前获取的不同单元字节；同一单元多次补载只记一次 |
| `candidate_precision` | 候选中真正出现在下一窗口 miss 集合的比例 |
| `probe_demand_h2d_bytes` | exact probe 自己引入的补载；resident 必须为 0 |
| `probe_workspace_d2d_bytes` | exact shadow 全部行的临时历史 KV 复制；不计入 H2D 或预取命中；resident 为 0 |
| `next_unavailable_units` | 下一窗口使用、但源窗口尚未成为已提交历史的单元 |

没有用 kernel 调用次数代替搬运量；实际 miss 来自 swap kernel 输出的 `miss_src/miss_count`，再映射回逻辑 C4 ID。

16K/32K/64K/128K 原始输入分别约对应 4096/8192/16384/32768 个 C4 单元，本轮共同常驻容量为4096。
历史超过容量不保证每个窗口都有 miss；若分母为零，指标输出 null，不伪造 100% 覆盖率。
若另外扩大 W 或 K，常驻安全下界也会上升；若调整容量，基线和影子组必须一起重跑并区分配置。

**已有完整 trace 可以重新计算这些指标，但不能补出未生成的第 12 个 draft token。** 要测试本轮 G=12、W=6/7 新生成 token，需按新协议重新采集。若只是重新分析旧轨迹，可保留原 audit，写入新目录；新报告会按实际候选数标注 G/W，并记录原采集的计数口径：

```bash
python benchmark/oasiskv/trace/analyze.py \
  /srv/oasiskv/trace-v4/w6/probes/trace.jsonl.gz \
  --audit /srv/oasiskv/trace-v4/w6/audit.json \
  --output /srv/oasiskv/trace-v4/w6/analysis-all
```

这统计的是完整预测集合的潜在覆盖率，既没有执行在线预取，也没有把通用路径 `max_new_blocks` 当作候选预算。

## 这些结果不能说明什么

本轮是质量实验，**明确关闭 decode/prefill CUDA graph 和 overlap scheduler**，以便执行快照、可变宽度 shadow 和逐层观测。它不更改正常服务的 graph 配置。CPU 快照、GPU→CPU 观测和额外 forward 都会明显干扰时序，不能据此报告 TPS、TPOT、probe 的增量延迟或异步重叠收益；相应字段保持 null。

目前也没有把候选真正异步搬入下一窗口、模拟其间的淘汰、或观测传输完成时刻。因此 `potential_*` 只是预测潜力，`timely_miss_coverage` 和 `saved_critical_path_ms` 一律为 null。

下一步先看各层、各接受长度下 resident 对下一窗口**实际 misses**的覆盖，以及相对 exact 的变化；同时查看扩大 proposal/限制 V 后的接受长度变化。只有这些数据支持继续时，再实现线上融合 forward、KV 并集的安全生命周期和 graph 下的真实传输时序测试。

CPU 验证命令：

```bash
PYTHONPATH=python SGLANG_USE_CPU_ENGINE=0 python -m pytest -q \
  test/registered/unit/kernels/test_hisparse_scan.py \
  test/registered/unit/spec/test_oasiskv_trace.py \
  test/registered/unit/spec/test_oasiskv_trace_startup.py \
  test/registered/unit/spec/test_oasiskv_trace_workspace.py \
  test/registered/unit/spec/test_oasiskv_trace_sweep.py \
  test/registered/unit/spec/test_oasiskv_trace_coverage.py \
  test/registered/unit/managers/test_prefill_adder.py \
  test/registered/unit/layers/quantization/test_fp4_moe_routing.py
```
