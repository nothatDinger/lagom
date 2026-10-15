# Round 23：CTA4／CTA8 的 warp 区段延迟诊断计划

状态：**测试设计，尚未实现本轮插桩 kernel 和执行脚本。**
本文给出实现、实机执行与反馈规范。不能把旧 `run_h2d_grid.sh` 的整体计时
当成本轮区段数据，也不能把下文约定的产物名误认为已有命令。

## 1. 唯一要回答的问题

同一 B32/miss100 分散源，CTA4 改为 CTA8 后，每 warp 搬运量从25条降到12–13条，
为什么整个 copy kernel 没有明显加快？

需要区分：

1. 单条 copy 调用耗时变长，抵消了每 warp 工作量下降。
2. copy 区段累计时间下降，但其余循环区段或启动/结束分布恶化。
3. 少数 warp/CTA 拖尾，掩盖多数 warp 的改善。
4. 插桩本身改变行为，现有测量不足以回答。

**总 request 遍历次数翻倍不代表墙钟延迟翻倍。**
每 warp 仍遍历32个 request，新增 warp 可能在其他 SM 上并行执行。
不把重复遍历预设为根因，也不预设 host 通路已经饱和。

## 2. 固定条件和最小矩阵

只用物理 GPU4（98:00.0），CUDA_VISIBLE_DEVICES=4，CPU/内存绑定 NUMA1。
共享 Portable 注册（flags=1）的512MiB host pool，完整落在 node1。

沿用 round15 的 B32/miss100、584B/KV、每次1,868,800B、64个分散源 bank、
plan_stride=3072、buffer_slots=3072、随机目的槽位和 legacy load helper。
所有方法共享 GPU plan、counts、real、源 allocation 和目的 buffer。
同 bank 的任务及次序一致。CTA4/8 使用同一个 runtime-grid JIT 产物。

### 主矩阵：6组 host 配置

| variant | grid | 用途 |
|---|---|---|
| legacy | CTA4/1024、CTA8/1024 | 原生产 kernel，主要性能基线 |
| control | CTA4/1024、CTA8/1024 | 独立探针实现，编译期关闭全部计时/记录 |
| trace | CTA4/1024、CTA8/1024 | 同探针实现，开启区段计时和每 warp 统计 |

control 用来检查复刻 kernel、包装和编译变化是否已改变基线；
trace/control 用来检查插桩扰动。不能只比较 trace4 与 trace8 就下结论。

先不加入扁平 plan、D2、其他 CTA、其他载荷、GPU5、Rx 或新的缓存策略。
HBM 只用于初期正确性检查；主问题如仍不清楚，再决定是否增加 HBM 性能对照。
所有 host 性能组都使用同一种源类型。

### 运行状态

为解释旧 CTA 实验，复用 round15 的相同计算预热方式：每方法前250ms compute
预热、两次 graph 预热，计时区间内无矩阵计算。记录每波前后 NVML 状态。
不把当前 round21/22 的 copy-only 预热和1980MHz结果直接拼进这轮。
状态读取放在计时区间外，无 Rx 轮询、无 NCU、无 profiler 主计时。

无插桩 legacy 必须当轮重测；若 CTA8 本轮已稳定加快，先报告现象改变，
不能强行用本轮区段数据解释旧环境中的“无收益”。

## 3. 插桩边界：测代码区段，不虚构纯硬件时间

复刻 `hisparse.cuh::copy_cache_planned_kernel` 的 request/count 及任务分配逻辑，
调用同一个 `copy_miss_item<true,true,0>`，不改生产 kernel 默认路径。

建议记录两类区间：

1. **loop_span**：从进入 request 遍历前到全部 request 遍历结束。
2. **copy_call**：每次 `copy_miss_item(...src_row[m], dst_row[m]...)` 的前后边界。
   明确将源/目的索引求值包含在该区段，不能在未证明其读取已完成时把等待划到别处。

每 warp 累计所有 copy_call 的耗时，定义：

```
copy_sum = sum(copy_call_duration)
loop_other = loop_span - copy_sum
copy_per_kv = copy_sum / copied_kv             # copied_kv=0 时输出 null
```

copy_call 包含 plan 索引读取、C4 地址计算、host load、结果消费和 device store 指令。
loop_other 包含 request/count 遍历、广播、分支、游标更新、部分计时开销，以及可能的
调度/等待。**它们不是“纯PCIe时间”和“纯准备时间”。**

这两个区段在动态执行中交错；不存在先完成所有准备、再完成所有搬运的阶段。
即使区段时间可加，它们也只是插桩版本的执行区间划分，不是可独立相减的硬件成本。

### 每 warp 的记录

每个实际 warp 输出一条记录，字段至少为：

```
bank, CTA_id, warp_in_CTA, global_warp_id, SM_id
loop_begin_global_ns, loop_end_global_ns
copy_sum_ns, copied_kv, request_visits
```

使用可跨 SM 比较的 `%globaltimer`；不能把不同 SM 的 `clock64()` 绝对值混合排序。
保留原始整数时间戳，报告只显示相对于本次 invocation 最早起点的时间。
每个 bank/invocation 独占记录区，graph 中后续调用不能覆盖待分析记录。
记录 buffer 在 capture 外分配；每波开始前用哨兵初始化，在完整同步后回读。

计时调用采用具有编译器屏障语义的实现，并检查最终 SASS。
不能仅凭 `asm volatile` 或源码位置断言计时边界正确。
不要为了“等待搬运彻底完成”在测量区段加入 `__threadfence_system()`、
CTA barrier 或额外跨 warp 同步：这会引入原 kernel 没有的串行化。
若不用这些手段无法可靠限定区段，应降低判读范围或报告此测法不可用。

尽量只在循环内累计寄存器统计，结束时由 lane0 写一条全局记录。
不用循环内 printf、全局 atomic 计时累计或逐条全局日志。
全局记录写出及最后 store 的排空不一定在 loop_span 内；完整 kernel 计时必须保留。

## 4. 第一阶段：正确性、codegen、计时语义

### 4.1 正确性检查

复用已有字节 oracle 和动态 graph 测试，覆盖三个 variant、CTA4/8、host/HBM：

- eager 及同一 graph 修改 counts/real/plan 后重放；
- real=0、零 miss、count=1/65/100/127/128/129，以及跨32请求窗口的混合请求；
- C4页边界、非连续 plan 行、padding request、目的未触碰字节和页 padding；
- stats guard、哨兵完整覆盖、warp/CTA编号唯一且范围正确。

分别从CPU任务分配计算实际预期记录数，不硬编码与测试矩阵不一致的通过条数。
每个 invocation 校验：

| B32/miss100 | CTA4 | CTA8 |
|---|---:|---:|
| warp记录数 | 128 | 256 |
| 每warp request_visits | 32 | 32 |
| 每warp copied_kv | 25 | 12–13 |
| copied_kv总和 | 3200 | 3200 |
| 每CTA copied_kv | 800×4 | 416×4、384×4 |

要求 loop_end>=loop_begin、0<=copy_sum<=loop_span；计时过粗或零耗时须明确标记，
不能用截断负值、添加epsilon等方式制造通过。

### 4.2 离线提取实际加载产物

记录 legacy/control/trace 的真实 `.so` 路径、SHA256、REG/STACK/LOCAL/SHARED，
使用已有 cuobjdump 的 `--dump-sass` 和 `--dump-resource-usage`。
不使用本机不支持的 `--demangle`，无需安装新版 NCU。

人工核对：

- control 无时间戳和stats写出，任务遍历及 copy load/store 形状与 legacy 一致；
- trace 时间戳、copy load/store、回跳和stats写出的完整路径；
- copy 的结果消费确实处在 copy_call 结束时间戳之前，没有跨区段调度或提前等待；
- 输出记录的写入在 loop_end 时间戳之后，且实际 globaltimer 读取未被消除；
- 无 spill；记录寄存器增加及可能的驻留资源变化，不把无spill等同于无扰动。

必须查看机器码完整控制字段和跳转辅助块；操作码计数只能作为提示。
若未能确认边界，标记 `timing_semantics_reviewed=False`，不能发布区段归因。

**这一阶段结束先反馈 check-compact.txt 和 codegen-compact.txt。**
原始 SASS 留本地，只按需补充连续片段。不要自行把 codegen 提取成功当成人工判读通过。

## 5. 第二阶段：插桩扰动检查与正式测量

在正确性与计时语义通过后，运行6个配对波次。每波遍历同一64-bank集合，
每个 variant/grid 都有64个 copy；波次为统计单位，不能把所有 warp 当独立样本。

平衡 variant 顺序和 grid 顺序：

| wave | variant顺序 | 每variant的grid顺序 |
|---|---|---|
| 0 | legacy, control, trace | 4,8 |
| 1 | trace, control, legacy | 8,4 |
| 2 | control, legacy, trace | 4,8 |
| 3 | trace, legacy, control | 8,4 |
| 4 | legacy, trace, control | 4,8 |
| 5 | control, trace, legacy | 8,4 |

每方法的预热、事件节点、bank 顺序和重放次数一致。保留无 profiler 的 CUDA
Event 每copy时间和整波墙钟时间；用同一口径计算1,868,800B/次的有效吞吐。
区段样本必须属于这些被计时的 trace invocation，不能拼接另一次运行。

插桩扰动作为前置门禁：

- 每种grid的6波中位 `control/legacy` 在0.97–1.03内；
- 每种grid的6波中位 `trace/control` 在0.95–1.05内；
- 每variant至少5/6波满足相应区间，避免仅均值抵消；
- `[(trace8/trace4)/(legacy8/legacy4)]` 的6波中位在0.97–1.03内，
  且至少5/6波满足。保留逐波值和所有失败项；
- 无错误、spill、记录覆盖问题或明显运行状态分化。

这些阈值是投入筛选条件，不是统计显著性检验，也不证明局部区段完全不受影响。
**差异若只有几个百分点、与插桩影响同量级，不据此归因。**
门禁失败保留完整平衡序列，输出 `segment_results_valid=False`，不自动放宽阈值或重跑。
此时 legacy 总体性能仍可单独报告，不能用 trace 区段解释 legacy。

## 6. 如何分析，避免错误相加

首先按每 invocation、每 warp 原始数据计算，再汇总各波分布。
报告 p50/p95/max 以及 copied_kv=12/13/25 分组；不要比较不同工作量组的原始
copy_sum 而忽略项数，也不要把分位数相减计算另一个区段的分位数。

分别统计：

- loop_span、copy_sum、loop_other、copy_per_kv 的分布；
- 每CTA实际SM_id、最早warp开始与最晚warp结束，均相对本 invocation最早起点；
- 全部warp开始时间跨度、全部warp结束时间跨度、活跃SM数；
- 每 invocation 最后到达 loop_end 的warp，以及它的copied_kv和三个区段时间。

不同warp的区段通常重叠，**不能将它们求和与kernel墙钟作分解**。
最后到达loop_end的warp只是拖尾线索，stats写出和store排空仍可能延后kernel结束。
不能把kernel总时间减去其loop_span称为准确launch延迟。

| 观察 | 支持的解释；不支持的过度结论 |
|---|---|
| CTA8每KV copy_call变长，copy_sum未按项数下降 | 搬运调用区段的有效处理速度下降；尚不能确定PCIe队列或TLB |
| copy_sum下降而loop_other增加 | 其他循环执行/等待可能抵消收益；不是已证明counts读取慢 |
| 多数warp更快、末尾少数warp/CTA不快 | 检查启动偏斜、任务分布、SM放置和拖尾 |
| 所有区段都更快但完整kernel没变 | 检查区段外工作、stats开销、计时语义与配对，不直接宣称新硬件瓶颈 |
| legacy CTA8已稳定更快 | 原“无收益”前提未在当前运行复现，先解释运行条件差异 |
| 插桩或语义门禁失败 | 此探针不能回答归因问题；停止机制判读 |

## 7. 实机操作顺序与最小反馈

1. 正常仓库属主同步分支，读取本文。实现端先新增独立探针和包装，不改服务默认路径。
2. CPU检查统计解析、任务覆盖、失败报告，以及禁止错误输出结论的门禁。
3. 在一个仅物理GPU4的gpuq作业内完成第4节正确性和产物记录，离线提取SASS。
4. 回传下面的第一份紧凑反馈，完成计时边界核对后再进入第5节；不要求提前跑性能。
5. 第二阶段一次作业完成全部平衡波次和扰动检查，回传第二份紧凑反馈。

**当前文档不提供尚未实现的启动命令。** 新脚本准备完成后，必须补上实际存在、
经过CLI/错误路径检查的命令；不要让执行端猜参数或直接复用旧grid入口。

第一份反馈，约10行：

```
CHECK success=... correctness=通过/预期 graphs=通过/预期 bytes_padding=...
IDENTITY physical=4 BDF=98:00.0 HOST node1=... register_flags=1
BINDINGS same_plan_source_target=... only_grid_changes=...
RECORDS CTA4=128 CTA8=256 unique=... guards=... counts_match=...
CODEGEN artifacts=3/3 extraction_success=... timing_semantics_reviewed=False
RESOURCE legacy REG=... STACK=... LOCAL=... SHA256=...
RESOURCE control REG=... STACK=... LOCAL=... SHA256=...
RESOURCE trace REG=... STACK=... LOCAL=... SHA256=...
```

第二份反馈，约20行，建议产物名 `latency-compact.txt`：

```
GATES correctness=... timing_semantics=... perturbation=... segment_results_valid=...
TOTAL variant={legacy,control,trace} CTA4_us=... CTA8_us=... ratios_8_over_4=[6波]
PERTURB CTA={4,8} control_over_legacy=[6波] trace_over_control=[6波]
PERTURB relative_grid_ratio=[6波]
SEGMENT CTA={4,8} kv_count_groups=... loop_span_us=p50/p95/max
SEGMENT CTA={4,8} copy_sum_us=p50/p95/max copy_per_kv_us=p50/p95/max
SEGMENT CTA={4,8} loop_other_us=p50/p95/max
PLACEMENT CTA={4,8} active_SM_count=... start_spread_us=... end_spread_us=...
TAIL CTA={4,8} last_warp_span/copy/other_us=... copied_kv=...
STATE ...
```

原始每warp记录、完整JSON、SASS和日志留在实机，不上传大文件。
失败给同一紧凑报告；报告未生成才给run.log最后30行。
本轮结果只决定是否值得继续扁平plan或其他执行组织，不直接改服务配置。
