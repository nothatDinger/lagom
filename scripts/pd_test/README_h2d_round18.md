# 第十八轮：相邻 KV gather 作为中间对照

分支 `perf/hisparse-h2d-diagnosis`，基于 hash_verify。
只回答：**保留原 gather 的每 KV 处理，把源和目标 KV 都放相邻，能恢复多少连续 SM 基线的性能？**

## 1. 三组，均 CTA4/1024

| 方法 | 源 / 目标 | 保留的处理 |
|---|---|---|
| gather4 | 随机 C4 源 / 原分页稀疏目标 | 原 legacy kernel、plan/count、每 warp 一条 584B KV |
| gather_adjacent4 | 相邻 C4 KV 源 / 相邻 C4 KV 目标 | **同一个 legacy kernel/JIT**、相同 plan shape/stride、共享 counts 和目标 allocation、原 value/scale 处理 |
| linear4 | 连续字节源 / 连续字节目标 | 第17轮连续 SM kernel，不改它的读取策略 |

全部 B32/miss100，每 copy 3200 KV = 1,868,800 B，64 banks。
相邻 gather 的源索引为 `bank_start + request*100 + miss`，目标索引为 `request*100 + miss`。
这两组 plan 的**索引内容改变**，plan/count 的读取、遍历逻辑及数组 shape/stride 均保留。
跨请求边界也相邻；C4 每页 64 KV，576B value 与 8B scale 仍分离，页 padding 保留。

相邻组复用既有双 plan 替换路径（内部 method 名为 sm_ordered，但本轮不执行排序），
其 Python load_policy="legacy"（JIT策略0），最终调用原 copy_cache_planned_mla；不调用 iterator、不改 CUDA 文件。
两种 gather 使用同一个 device output allocation，保持其容量，只改变写入位置。
目标索引是 copy kernel 支持的绝对 slot；本轮没有接 attention，不能将该目标重映射直接用于服务。

## 2. 数据与计时约束

物理 GPU4/5，单进程共享 NUMA1 的 512 MiB host allocation；注册 context=CUDA1、Portable flags=1。
同一 allocation 内预留三段互不重叠区域：相邻 C4 banks、连续字节 banks、原始随机 C4 源。
在 CPU 上复制 value/scale，连续组按 KV ordinal 的 `[576B value,8B scale]` 排列。
**三组每 bank 的 KV 内容相同**；CPU 对全部 64 bank 检查 payload，并确认原随机源未被覆盖。
packing/plan 构造均在计时外，耗时只单独记录，不冒充服务收益。

为同时保留三份内容，随机源地址范围约 297 MB，比第17轮小。使用本轮同时采集的 gather4，
不跨轮相减；区域大小、源 footprint 细节留在 bridge.json。虚拟页数不代表 TLB/物理事务。

两卡每组先检查 bank0/63 的全部字节；gather 检查未触碰 slots/页 padding，linear 检查两侧 guard。
两卡三组最终持续 graph 均 reset/replay/同步/校验，共 6 个。
绑定检查 GPU 侧实际 src/dst 内容、shape/stride、共享 counts/output、参数、pinned host 及每 chunk 工作量。
记录实际加载的 legacy/linear JIT 路径和 SHA256；三个方法每卡使用相同 bank 序列重复次数。

使用已验证的 event-free chunk graph，每段只提交一次；每窗口 copy 预热2秒、测量2.5秒，裁掉首尾0.3秒。
不跑 matmul 预热；本轮不采 Rx、不开 profiler、不运行 CE、不加 CTA。
设备顺序 `[0,1,1,0]`，每卡第一轮 `gather4 → gather_adjacent4 → linear4`，第二轮反序。
共12个窗口，约30秒正式测量+24秒预热，另有CPU准备/JIT/检查时间。
NVML 状态仅在窗口外读取；端点状态不能证明窗口内恒频。

质量门不放宽：提交/GPU≤0.10、GPU/wall=0.90–1.02、chunk CV≤0.10、
窗口后段/前段耗时0.90–1.10；每方法两轮归一化耗时也须0.90–1.10。
字节/绑定/CUDA异常立即失败；窗口质量失败保留其余平衡数据，结果无效，不输出 RATIOS。

## 3. 实机执行：check 通过后 main

正常属主账号在仓库根目录同步：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

在现有 gpuq 分配物理 GPU4/5 的作业内，使用现有 venv：

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_kv_bridge.sh check "$PWD/work/h2d-round18/bridge-check"
```

check 不计性能；期待 `success=True mode=check correctness=6/6 correct=True graphs=6/6 graph_valid=True packing=True bindings=True JIT=True results_valid=True`。
同时核对 identity physical4/5、host NUMA1。通过后直接继续；失败仅反馈错误，不进入 main。

```bash
bash scripts/pd_test/run_h2d_kv_bridge.sh main "$PWD/work/h2d-round18/bridge-main"
```

不安装工具、不改驱动、不需要 Docker/NCU。main 再做全部检查，期待上述布尔项全部通过及
`TIMING valid=True windows=12/12`。本地环境无 GPU/nvcc，未声称已验证实机 CUDA 执行。

## 4. 只反馈这些

- check 的通过行一条，并确认物理卡/NUMA正确。
- main 的 bridge-compact.txt 全文，约20行；原 bridge.json 留在实机。
- 若失败，给该 compact；没生成时只给 run.log 尾30行，不重新跑或自行放宽门。

RATIOS 每卡两轮均为**耗时比**：adjacent/random、linear/adjacent、linear/random。
两轮方向一致且约≥10%的变化才推进归因。
相邻组接近 linear，支持布局/footprint 恢复了本次主要差额；不能拆成单独的源、目标或 TLB 收益。
相邻组仍落后，剩余差额综合了 plan 遍历、C4分离、每warp有效字节和编译差异；不能直接宣布其中某项主导。
先得到这个中间点，再选择下一项；本轮不扩大参数扫描，也不推服务 packing。
