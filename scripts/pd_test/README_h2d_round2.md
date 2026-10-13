# 第二轮：区分 copy 并行度、NUMA 和四卡共享资源

分支仍是 `perf/hisparse-h2d-diagnosis`，以 `hash_verify@0fef460d` 为基础。
原有工作区和生产 copy 参数保持原样。本轮使用原生 Python/venv，不需要 Docker。

## 1. 首轮已经支持的判断

以下数值来自用户 2026-10-13 上传的报告；尚未拿到原始 trace/probe/JSONL，属于汇总核对。

| 模式 | TP0 有效载荷 GB/s | TP1–3 有效载荷 GB/s | 四 rank 每卡累计载荷 |
|---|---:|---:|---:|
| C，正常 residency | 30.26 | 22.53–22.59 | 约 2.49 GB |
| D，强制 H2D | 34.72 | 24.39–24.46 | 约 31.93 GB |

- C 的 B=32/W=6 完全一致，四 rank 的载荷相差不到 0.1%，而慢 rank 的累计 copy 时间比 TP0 多约 34%。
  D 的这个时间差扩大到约 42%。优先核查实际 GPU、host 页位置、运行上下文，不能用一个全局 CTA 参数解释全部差异。
- D 载荷增加约 12.8 倍，带宽仅增加约 8–15%。不支持把问题全部归结为小 copy 的固定启动成本。
  D 包含 B=30/31/32，需用原始分析的 `geometry`/`captured_geometry` 分组后再比较。
- 首轮 trace 已经得到 copy 与 recorder 各自的重叠统计，均近似零。多个 stream ID 不代表并发。
  可以先离线检查已有文件，暂不为这个问题单独重跑 Nsight。
- A2/A3 的 TPOT 均值 7.514 ms，B2/B3 为 7.4945 ms：未激活 recorder 时未见明显恶化。
  这两次重复不提供统计显著性，也不衡量激活 recorder 和 profiler 的完整开销。
  C 的 recorder 自身时间约占 copy 时间 2.6–3.9%，不能直接当成 decode 的边际开销。
- A1/B1 的 TTFT 约 9.5 s，后续约 1 s。启动状态和重复 seed 导致的 prefill radix-cache 命中都可能影响它；
  第一轮 TPOT 实际低于后续轮次，不能仅凭这些数据称为“冷启动 TPOT 更慢”。

代码证据：

- `python/sglang/srt/managers/hisparse_coordinator.py`：`_init_shared_index_prefetch` 设 `_prefetch_copy_blocks=4`，
  `_run_swap_in_verify_window` 的同步 DSPARK copy 复用这个值。
- `python/sglang/kernels/ops/kvcache/hisparse.py`：`copy_cache_planned_mla` 默认 4 CTA、每 CTA 1024 threads；
  注释说明小 grid 原本用于 side-stream 与计算重叠。
- `python/sglang/kernels/jit/csrc/hisparse.cuh`：`copy_cache_planned_kernel` 用 warp 遍历 miss plan，直接读 host 地址。
  这是 SM 发起的稀疏搬运；本路径没有 CPU copy-worker 队列。4 CTA 最多占用 4 个 SM，是否限制在途请求需实测。
- `python/sglang/srt/mem_cache/memory_pool_host.py`：C4 host KV 按层分别 mmap，再 `cudaHostRegister`。
  `python/sglang/srt/utils/numa_utils.py` 的自动 NUMA 配置可以从硬绑定退回 preferred、CPU-only 或跳过。
  GPU 的 NUMA 归属、允许使用的内存节点，都不能替代实际 host 页归属。

原始文件还需要核对 layer/verify 边界、rank 身份和 acceptance。
W=6 是候选验证宽度，不等于每次实际提交 6 个 token。不能用 copy 时间直接预测 TPOT 的优化比例。

## 2. 先保存现有进程的 NUMA 证据

在现有实验 worktree 更新分支；先确认没有本地 tracked 修改，再执行 fast-forward：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R2="$LAGOM_REPO/work/h2d-round2"
mkdir -p "$H2D_R2"

"$LAGOM_PY" scripts/pd_test/collect_h2d_numa.py \
  --probe-dir work/h2d-round1/normal-trace \
  --output-file "$H2D_R2/live-numa.json"
```

这一步不会加载模型、搬运 KV 或重启服务。旧 probe 没有进程启动标识，PID 会标为未验证；
用结果中的 `nvidia-smi --query-compute-apps` 确认是当前 decode worker。
如果已重启，改用 `--pid PID0 PID1 PID2 PID3`，不要将旧 PID 的读数归给旧 trace。
若是别的 Unix 用户启动的服务，读取可能被拒绝；输出保留权限错误。只需让服务所属用户运行这个只读脚本。

旧版本只支持进程整体 NUMA 分布，不能精确归因到 KV。
以后**从新版代码重启并采集 C**，`.h2d.json` 自动新增 GPU UUID、CUDA_VISIBLE_DEVICES、进程 start_ticks、
各层 C4 host buffer 地址和 `/proc` NUMA 快照。这些 CPU 元数据仅在初始化和采集结束时处理，decode 热路径不增加操作。
此刻不要求仅为元数据重启服务。
VMA 的 node page counts 是整段映射的统计；`buffer_covers_whole_vma=false` 时不能按大小比例分摊。

## 3. 独立 copy 实验：先 smoke，再单卡和四卡

先保存首轮文件和上述进程证据，然后通过你现有的 gpuq/服务管理方式停止实验服务、释放 GPU。
以下测量要求 GPU 4–7 空闲，并避免其他大流量 host-memory 工作；共享 DRAM 会影响结果。
脚本不会停止进程、改变系统 NUMA 策略或修改驱动设置。

先做一次功能 smoke，检查生产 kernel 数据正确性和图内 event 计时：

```bash
CUDA_VISIBLE_DEVICES=4 "$LAGOM_PY" scripts/pd_test/bench_h2d_copy.py \
  --batch 2 --misses 2 70 --blocks 4 8 --source-mib 4 --max-banks 4 \
  --bulk-ce-mib 1 --rounds 2 --warmup 1 \
  --output-dir "$H2D_R2/smoke" >"$H2D_R2/smoke.log" 2>&1
```

必须退出成功，`smoke/rank0.json` 中 `success=true`，全部 correctness 通过、copy 时长为正。
smoke 的工作集很小，其带宽不用于性能判断。失败时先返回日志与部分 JSON，不继续矩阵。
CPU 开发环境已验证解析、几何计数和独立 CPU 字节校验；GPU event/capture 仍需这个实机检查。

确认 `numactl --hardware` 的 node 1 对应 decode GPU；本机报告为 node 1。使用**硬内存绑定**，
绑定失败则停止，不静默回退成另一种 NUMA 配置。以下四个单卡命令顺序执行：

```bash
for gpu in 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$gpu" numactl --cpunodebind=1 --membind=1 \
    "$LAGOM_PY" scripts/pd_test/bench_h2d_copy.py \
    --output-dir "$H2D_R2/solo-gpu${gpu}-node1" \
    >"$H2D_R2/solo-gpu${gpu}-node1.log" 2>&1 || break
done
```

检查四个 JSON 全部成功，再做四卡并发：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 GLOO_SOCKET_IFNAME=lo \
  numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/pd_test/bench_h2d_copy.py \
  --output-dir "$H2D_R2/four-gpu-node1" \
  >"$H2D_R2/four-gpu-node1.log" 2>&1
```

默认每卡用同一个 512 MiB mmap/register host 分配，固定 B=32，100 和 1272 misses/request，
有效载荷分别 **1,868,800 bytes** 和 **23,771,136 bytes**。
它们近似首轮 C/D 的平均 copy 大小；是均匀 synthetic 控制，不是实际 miss plan 的逐条重放。
默认扫描 CTA=4/8/16/32/64，threads 固定 1024，同时测相同 byte 数的连续 H2D copy，另加 64 MiB bulk 对照。
miss-plan 的行 stride 默认 3072，与 W=6/top-k=512 的生产缓冲一致；destination 在每请求 3072 slots 中随机选择，
请求之间留 64-token page padding。两项可用 `--plan-stride`/`--buffer-slots` 显式覆盖。

每项比较复用同一组随机、不重复的源地址，源数据填充随机字节，使用多个不重叠 bank 轮换；
正常配置轮换有效载荷超过 100 MiB，降低固定小块源地址反复命中 GPU cache 的影响。
它不能替代 PCIe wire counter，也不能证明所有访问都没有 cache hit。
每个 CTA 配置的首尾两个 bank 会校验全部 value 和 scale 字节。
输出、分配、校验、miss plan 上传、JIT、Gloo barrier 均在 copy 计时之外。
计时 events 是 CUDA Graph 内的显式 record 节点，逐次包围单个 copy，避免 Python 提交间隙。
graph 中的 events 本身可能影响很短的 copy；它是统一的测量口径，不能把 event 时长强行当成首轮 trace 的 kernel 时长。
报告使用 `total_payload / sum(copy_time)`；不平均逐次 GB/s。
测试顺序按固定 seed 打乱，各 rank 顺序一致。并发测试每轮用 barrier 加一整轮 bank copy；
保存各 worker 的 host 提交/完成区间和 GPU wave 时长，便于检查是否明显错开。
barrier 本身不保证跨 GPU 同时开始，不能凭 rank 数量虚构精确重叠或八卡总线带宽。

每次完整运行仅是一次实验重复；默认 rounds 是运行内重复，不代表独立服务重启。
发现差异后再用新 output-dir 复测相应项，避免先做过大的矩阵。

## 4. 如何根据结果选择下一步

| 结果 | 支持的方向与下一步 |
|---|---|
| 单卡本地内存下，增加 CTA 明显提升 SM 带宽，CE 不变 | 4 CTA 限制 copy 并行度；再将候选 CTA 用于正常 residency 服务 A/B，核查 TPOT、acceptance 和其他 kernel 竞争 |
| 单卡接近，四卡同时运行明显下降；CE 也下降 | 优先查同 NUMA 节点的 DRAM/PCIe 共享资源及实际并发，不能只加 CTA |
| 固定本地内存时同一物理卡始终更快，单卡也如此 | 优先核对实际 UUID/PCI bus、GPU/PCIe 状态；“TP0 角色”不足以解释独立实验差异 |
| 独立实验四卡接近，服务仍有 TP0 分化 | 优先查服务真实 host 页、地址分布、GPU 其他工作/同步；需要实际 plan 或 rank 与物理卡换位对照 |
| CE 较高、SM 在足够 CTA 下仍低 | 继续查稀疏请求的 transaction 放大、scale/value 访问、读延迟与在途请求，不能把 CE 上限当成相同访问模式的必达值 |

只有数据指向 NUMA，或需校准远端 penalty，才追加远端内存对照：
复制四卡命令，保持 `--cpunodebind=1`，把 `--membind=1` 改为 `--membind=0`，
结果放到新的 `four-gpu-node0-memory` 目录。这样改变内存位置时保留 CPU 位置。
先确认 NUMA0 允许使用、内存充足；失败就保留错误，不修改系统策略。

## 5. 请返回的数据

先返回 `live-numa.json`，以及首轮 `analysis-C/`、`analysis-D/`、六个 A/B JSONL、
environment JSON、对应 decode 日志的 `accept len`/`avg_spec_accept_length` 数据。
分析 JSON 保留 sample/interval 明细；另保留原始 `.h2d.json` + `.trace.json.gz`，可打包上传以便核查配对与时间线。
仅有汇总 Markdown 不能完成原始数据审计。

独立实验返回整个 `work/h2d-round2/`：smoke/单卡/四卡的 rank JSON 与日志。
JSON 已包含逐次 GPU copy 时间、每波区间、NUMA 分配证据、配置和 GPU 身份。
正常服务的新 C 采集、远端对照和生产 CTA 修改都由这些结果决定，本轮先不要求重跑整套 ShareGPT。
