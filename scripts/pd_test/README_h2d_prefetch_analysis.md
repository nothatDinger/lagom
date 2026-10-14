# 跨 KV 读取分析：先核对依赖，再决定实验

分析对象：`3e6cf38f3` 的 CUDA / DSv4 verify copy 路径，默认 legacy、CTA4×1024。
本次增加 CPU 遍历核对脚本和分析记录，没有修改 CUDA kernel 或服务设置。

后续实现见[第十三轮：D1/D2正确性与编译产物检查](README_h2d_round13.md)。
下文保留第十二轮分析结论；遍历核对脚本现已加入两个候选的CPU模型，仍不预测带宽。

## 1. 决定与证据边界

**源码和旧SASS均确认存在可改变的跨请求读取组织。值得准备一个深度1/2的受控候选；本次不启动 GPU 性能实验。**
目标是检验：在同一分散地址集合、同一128个 warp 下，让每个 warp 提前读取下一条 KV，
是否降低搬运耗时。正式性能比较之前须确认编译产物确实增加了跨 KV 的重叠，且正确性通过。

已确认算法依赖、源程序组织和同warp跨KV的指令顺序，不是实际在途事务数、
硬件队列深度或带宽根因。用户补充了第九轮旧
`vector-smoke-codegen/legacy.sass.txt` 的source load/store及回跳片段，
与之前三条64-bit source load、三条64-bit store和34寄存器/无spill摘要一致。
此核对不需要NCU。本地没有对应CUDA编译器或真实gather cubin，二进制证据来自用户提供的旧提取结果。

## 2. 哪些是真依赖，哪些只是当前组织

代码入口：

- `python/sglang/srt/managers/hisparse_coordinator.py::_run_swap_in_verify_window`
  （约2052–2145行）：先 `plan_cache_window_hash_mla`，再调用 `copy_cache_planned_mla`。
  当前路径顺序提交，plan 已生成后才执行 copy；copy 不等待 CPU 生成后续 KV 的地址。
- `python/sglang/kernels/jit/csrc/hisparse.cuh::copy_cache_planned_kernel`
  （约889–942行）：按 request 遍历，将扁平 miss 序列轮转分给各 warp。
- `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh::transfer_item`
  （约68–115行）：计算地址，读取前512B、尾部64B和scale8B，再写到目的位置。

对一条合法 miss，依赖关系为：

```text
GPU counts / warp编号 / 累积start → (request, miss)
                                  ├→ GPU src plan → host地址 → value/scale → GPU store
                                  └→ GPU dst plan → 目的地址 ──────────────────┘

下一条(request, miss)只依赖counts、索引与warp编号，不依赖上述value/scale。
```

因此 host load→消费其结果的 store 是必要的数据依赖；将下一条 src plan/host load
放在当前 KV 的 store 之前，在合法 plan 的约束下是允许的。
当前函数只保存一条 KV 的值，没有显式跨 KV 暂存或预取。copy kernel 内没有
`__syncthreads`、CPU队列等待或每KV的CUDA stream同步；`__shfl_sync`用于分发counts。
源码没有要求下一条 load 必须等上一条 store **完成写入HBM**，不能把源码顺序解释成该硬件行为。

一个 KV 内已经有并行空间：每 lane 读取两份64-bit前缀值和一份64-bit尾部/scale，
源码把这些值暂存后再store。第九轮SASS显示三条source load排在三条依赖store之前；
它们之间插有地址计算，单KV内存在重叠空间。128个warp也同时提供延迟隐藏。
因此“一次循环处理一条KV”不等于“整个kernel只有一笔在途读取”。

用户提供的关键指令顺序如下（省略地址计算，不省略本函数已有统计中的source load/store）：

```text
0640 LDG.E.64.CONSTANT R30, desc[UR6][R12.64]
0690 LDG.E.64.CONSTANT R12, desc[UR6][R12.64+0x100]
06a0 LDG.E.64.CONSTANT R14, desc[UR6][R14.64]
...  计算目的地址等独立指令
0840 STG.E.64 desc[UR6][R18.64],       R30
0850 STG.E.64 desc[UR6][R18.64+0x100], R12
0860 STG.E.64 desc[UR6][R28.64],       R14
0870 @!P1 BRA 0x520   # 同request的下一条miss
0880 BSYNC B0
...  request游标更新
08d0 @!P0 BRA 0x1c0   # 下一个request
...  32-request窗口更新
0900 @P0 BRA 0xf0
```

`STG`消费相应64-bit寄存器对：R30:R31、R12:R13、R14:R15。
沿这条执行路径，同warp下一条KV的source load位于这三个store之后；每条store
至少要等自己的源load结果就绪才能发射。因此，当前实现没有在当前KV的source load
仍未返回时，就发起同warp下一条KV的source load。
这是可直接读出的跨KV限制，不需要把机器码字段猜成动态等待时长，也不需要NCU来确认控制顺序。
`BSYNC`是线程分支重汇合指令，不能解释成所有CTA或PCIe传输的全局屏障。

这里无需假定store已写完HBM：下一KV的load可能与上一KV尚未完成的store重叠，
这和两条KV的host load组是否重叠是不同问题。
未确定的是此限制在128个warp并发下损失了多少带宽，仍需受控干预测量。

安全前提：plan/counts在本次copy期间稳定、源内容已就绪、源/目的allocation分离、
每条miss使用合法槽位。hash planner为去重miss分配并保护victim；不同request使用各自槽位。
现有kernel已经允许不同warp并行copy，不应在候选中扩大别名/生命周期假设。
读取前瞻必须受 `num_real_reqs`、各行 `miss_count` 和plan stride约束，尾项及padding不能越界。

## 3. 关键发现：目标载荷的预取必须跨 request

默认 `total_warps=4×1024/32=128`。原循环为：

```cpp
for (each request r) {
  m0 = positive_mod(warp_global - start, 128);
  for (m = m0; m < miss_count[r]; m += 128)
    copy_one_KV(src[r, m], dst[r, m]);
  start += miss_count[r];
}
```

每个warp实际持有扁平序列 `w, w+128, w+256, ...`，但代码逐行找到属于该warp的项。
**miss100小于128，同一warp在一个request的内层循环最多执行一次。**
只在 `m` 循环内unroll、提前读 `m+128` 或加双缓冲，对这个目标用例没有第二条有效KV可读。

CPU模型将原循环与独立的“扁平下标→前缀和二分查找request”oracle比较：

```bash
python3 scripts/pd_test/analyze_h2d_copy_schedule.py
```

本地结果：4,204组核对通过，覆盖空行、不同miss数、31/32/33及63/64/65个request、
真实batch小于captured batch的padding，以及模运算边界。它验证遍历算术，不验证CUDA执行或预测带宽。

| 合成输入 | 总KV | 每warp KV数 | request内最大循环次数 | 相邻任务跨request次数 |
|---|---:|---:|---:|---:|
| B32 × miss100 | 3200 | 25 | 1 | 3072/3072 |
| B32 × miss1272 | 40704 | 318 | 10 | 3968/40576 |
| B32 × miss2（轻载示例） | 64 | 0–1 | 1 | 无后续任务 |

miss100下warp0依次处理 `(request,miss)`：
`(0,0) → (1,28) → (2,56) → (3,84) → (5,12) → ...`。
可用的下一条KV在后续request，需要跨行保存暂存值，并推进plan遍历。
真实服务各请求的miss数不均匀，上表不证明每个真实请求都少于128；它准确描述现有目标基准。
轻载示例也不是对历史服务负载的测量，不能据此声称原作者因低压力而选择深度1。

## 4. 可改变的部分和代价

候选可以保留相同的warp→任务映射，提供跨request的 `next_item()`，再分离load和store。
首个干预只需两项分组，不必立即实现复杂的多级流水：

```text
一项暂存：load(A) → store(A) → load(B) → store(B)
两项暂存：load(A) → load(B) → store(A) → store(B)
```

这里的load指完整KV的寄存器读取；store保留原lane分工及C4位置。
两项需要跨request，末尾只剩一项时单独完成。无需CPU打包、额外H2D或改变源布局。
候选不应通过 `prefetch.global` 的缓存提示代替实际两项load，也不应同时改变cache policy。
只提前读取GPU plan能隐藏部分metadata依赖，但不能当作提前读取host KV的验证。

暂存第二条KV的值本身需要每lane再保存3个64-bit值，即6个32-bit寄存器，
另外还有地址、游标及有效性状态；编译器可能复用或重排，不能按34+6预测最终寄存器数。
当前legacy实测34寄存器、无local memory。H100每SM有65,536个32-bit寄存器，
两个1024线程且每线程34寄存器的CTA已需69,632个（尚未考虑分配粒度），
所以增加少量寄存器并不必然把原来两个CTA/SM降成一个；基线已容不下两个。
但1024线程的block仍有资源上限，前瞻可能导致spill或额外指令，需要实际资源输出确认。
`__launch_bounds__(1024,1)`约束编译资源，并不提供host读取的完成/并行度保证。

旧源码明确解释了小grid的意图：减少SM占用，与side-stream计算重叠。
verify复用了这个copy kernel；该意图支持保留4个CTA考察warp内部重叠。
它不能证明“跨KV无显式暂存”也经过相同权衡或曾在原场景最优。

## 5. 最小实验应怎样隔离因果

旧SASS顺序核对已完成。当前值得准备下面一个三组对照，性能执行前须有候选codegen及正确性证据：

| 组 | 遍历 | 跨KV暂存 | 用途 |
|---|---|---|---|
| legacy | 原实现 | 原实现 | 已知基线 |
| iterator-D1 | 跨request游标 | 1条 | 隔离遍历重写/代码生成成本 |
| iterator-D2 | 同一游标 | 2条，先读后写 | 检验跨KV重叠的增量 |

固定分散源、GPU plan/counts/目的、legacy读取策略、CTA4/1024；以B32/miss100为主，
miss1272为补充，两种源host/HBM。沿用已验证的配对计时、正确性及小报告流程。
先只测深度2，不做深度/线程/CTA/布局的笛卡尔扫描。

必须确认：D2的实际SASS中第二条KV的host读取提前到了第一条KV的依赖store之前；
source load形状/策略没有意外更换；寄存器/local-memory信息完整。
如果候选编译后又串行化，则该候选不能检验假设。
正确性须覆盖跨request、跨32-request窗口、零miss、奇数尾项、非连续plan stride、
真实batch/padding变化的graph replay，以及value/scale/未写字节。

判断：D1/legacy说明遍历改动的影响，D2/D1说明增加暂存的影响，D2/legacy判断净收益。
只有目标小载荷净收益持续、HBM/状态/正确性对照有效，才值得进入真实decode验证。
D2无收益只能降低该候选的优先级；不能证明所有预取策略无效。
D2有收益支持读取组织限制当前性能，不能单凭它确定PCIe tag、TLB或物理队列是根因。
