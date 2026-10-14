# 第八轮：依赖读成本与独立读链的并行响应

继续在 `perf/hisparse-h2d-diagnosis`（基于 hash_verify）工作。
目标仍是正常驻留 B32/W6 的 decode H2D 和服务吞吐。第七轮排序不进入生产。
本轮提供一个独立 CUDA 诊断 kernel；生产 copy 参数和默认策略不变。
只使用空闲物理 GPU4/5、NUMA1、原生 venv，不需要 Docker 或重启 8 卡服务。

## 1. 第七轮闭合了什么，还有哪些限制

已确认大载荷八组原字段计时自洽，gap 为 4.187–4.303 µs；全部门禁通过。
同地址集合排序后，host 大载荷 activity 降低 5.69% / 3.25%，卡间比值
1.3582 → 1.3934。小载荷此前报告改善不到 1%，且排序成本没有计入。
准确结论是“这一次同集合升序未消除主要差距”；不能推广成任何访问顺序都无影响。

滚动简报还有两处旧比值，不应进入下一轮模型：

- HBM 大载荷 random activity 比为 `278.627409/272.230727=1.0235`，不是 1.015；
  ordered 为 0.9844。GPU4 的 HBM 排序耗时增加 3.02%，本轮没有证据称其只是噪声。
- 小载荷若沿用先前 activity `61.53/82.35` 和 `61.07/81.62`，卡间比应为
  1.3384 / 1.3365，不是旧 event 列的 1.320 / 1.324；本次未展开其原 JSON，
  不要求为此重跑 GPU，也不使用旧 event 列继续推断。

PCIe 只读检查的最终事实是：endpoint 的 MaxPayload=256 B、MRRS=4096 B，
两端 `10BitTagReq` 能力均支持但启用位均关闭；各级链路均 Gen5 x16。
**root 启用位存在一处真实差异**，不能写成“全部静态字段相同”：

| root 字段 | 95:01.0，GPU4 路径 | b5:01.0，GPU5 路径 |
|---|---|---|
| DevCap2 10BitTagComp / 10BitTagReq | + / + | + / + |
| DevCtl2 10BitTagReq | + | - |

该控制位允许该 requester 使用 10-bit Tag；GPU 发起 host read 时，GPU 是 requester。
root 的 requester 启用位不能直接等同于 GPU 的可用 Tag 数；两侧 root 的 completer
能力均支持。目前它是配置差异证据，尚无因果干预支持。不会据此修改 PCI 配置。
可参照 [Intel 的 DevCtl2 字段定义](https://edc.intel.com/content/www/us/en/design/publications/core-processors-series-3-i-o-registers/001/device-control-2-dctl2-offset-68_2/)；
这里只引用标准字段语义，不将该型号文档当作实机 root 的实现证明。

另一个字段解释更正：`ARICap ACS-` 表示 **ACS Function Groups Capability** 不支持，
不能据此说 endpoint 的 ACS 总开关关闭；真正 ACS 扩展能力需分别看 ACSCap/ACSCtl。
定义见 [pciutils header](https://github.com/pciutils/pciutils/blob/master/lib/header.h)。
未暴露 ATS/PRI/PASID capability，不证明 GPU 地址转换/TLB 或主机 IOMMU/IOTLB 没有开销。
本轮不把标准 ATS 正在工作作为前提。可见字段也无法穷举固件、映射和内部仲裁配置，
因此不能把剩余原因直接宣布为动态队列问题。

## 2. 本轮只测两个问题

1. 只有一条依赖读链时，GPU5 的 host 读取成本是否已比 GPU4 高？
2. 单链接近时，差距是否随着独立链数增加才出现？

`bench_h2d_reads.py` 复用共享 mmap、Portable 注册（context=1）、UUID 核对、NVML、
图内 external CUDA Event 和双卡 ABBA。每次 graph 同步完成后才切换 GPU。
每个活跃 warp 只有 lane0 读一个 8-byte next-index，下一次地址依赖刚读出的结果。
编译期禁止展开该循环；最后保存终点、全部 node ID 的和以及 SM ID，防止空跑通过。
所有配置使用同一编译出的 kernel，threads/CTA 固定 1024，active warps 是运行时参数。

| 控制量 | 正式运行 |
|---|---|
| 共享 host 分配 | 512 MiB，同一地址、同一 NUMA 位置 |
| node ID 集合 | 从 131072 个 ID 中固定随机选择 32768 个，各配置相同 |
| 每次 kernel 的总逻辑读次数 | 固定 32768，每地址恰好一次 |
| grid（CTA × 每 CTA 活跃 warp） | 1×1/4/16/32；4×1/4/16/32 |
| 独立链数 N | 1、4、16、32、4、16、64、128 |
| 每链依赖跳数 | 32768/N |
| node 间距 | 128 B、4096 B，同 stride 内源地址集合固定 |
| 源对照 | 同一 registered host；本卡 HBM 中的相同数据 |

链数变化时将同一 node 序列按 N 路交织分成不相交环，每环走完整一圈。
每个配置、每张卡、两种源均核对所有链的终点与独立 CPU 求和，以及输出前后保护行。
同时验证 eager 和清空输出后的 CUDA graph replay；每次 kernel 写独立输出区，
每个测量 wave 结束检查所有计时 kernel 的输出，包含每次的保护行与 SM ID。
固定 grid 下增加活跃 warp 与相同链数分布到不同 CTA 可分别比较，实际 SM ID 会记录；
CTA 数量本身不保证占用相同数量 SM。

为避免重复环退化成 GPU L2 命中测试，probe 使用 `ld.global.cv.u64`。
PTX 对 system memory 的 `.cv` 定义是丢弃匹配的 L2 行并重新取数，
见 [NVIDIA Cache Operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators)。
这是诊断 opcode，与生产 legacy load 的事务形状不同；没有预设它能复现生产的 35% 差距。
HBM 仍使用同一 kernel/opcode，但 `.cv` 的 system-memory 行为不等同于 HBM cache 行为。

计时与判读边界：

- `event_us` 是图内一对 CUDA Event 夹住单个 kernel 的时间，没有 profiler/activity 列。
  报告使用无 profiler 的 event 数据；单次多跳摊薄启动成本，不需要另外解析 trace。
- N=1 的 `hop_ns` 为 kernel 时间除以依赖跳数，包含循环、地址转换及整个读路径；
  它不是纯 PCIe RTT。N>1 的该值还包含争用/调度，不能称为单次物理请求延迟。
- `Mloads/s` 是逻辑 8-byte load 数/时间。**不报告 PCIe GB/s，不用 N 当硬件队列深度**，
  也不使用带宽×延迟反算硬件 Tag 数。
- stride 对照改变虚拟页足迹，也改变地址局部性和 DRAM 访问；若有差异只能先称足迹敏感，
  不能独立证明 TLB。报告中的 4KiB page count 是虚拟地址统计，不是硬件转换页大小。
- HBM 时间不从 host 时间中扣除。单链/多链结果用于选择下一步，尚不构成服务优化。

## 3. 先 smoke

### 首次 smoke 失败与修复

旧版本 `1d908a739` 的读探针调用 `capture_wave` 时没有传入 stream。
PyTorch 的 `torch.cuda.graph.default_capture_stream` 是类级复用的 stream，
首次在逻辑 GPU0 建立后，切换逻辑 GPU1 不会自动重新建立。
已有双卡 copy 探针显式传入每设备 stream；新读探针遗漏了这一点。

旧 `reads.json` 已确认：stride=128、grid=1×1、device=0 的 host/HBM 两项均通过
eager 和 graph；接下来 device=1 的 host eager 通过，失败在 `graph_check`。
因此 `correctness=2/16` 是两项完整通过、第三项中断、其余十三项未执行，
不能解释为十四项都失败，更没有执行到 4×32。此序列与跨设备默认 stream 缺陷吻合。
本地 CPU 回归通过真实 PyTorch graph Python 上下文、模拟 CUDA 原语复现了
旧调用在 device=1 时仍进入 device=0 capture context 的行为；这不是 GPU 运行验证。

修复为每项创建并保留明确属于目标设备的 capture stream，在 FFI launch 前核验
当前设备和 stream，在 replay 前核验全部 Event 的设备。
JSON 记录每项 `capture` 校验；失败时 compact 输出 `FAILURE_CONTEXT` 与已完成的
`COMPLETED` 项，区分 eager、capture、graph replay/check、测量阶段。
kernel、`.cv` 指令、地址集合、链布局和计时方式均保持原样。
修复现已由下面记录的实机 smoke 验证；旧失败运行没有可用的性能结论。

`expected=[24,32640]` 的含义是 `[终点 node ID, 访问的 node ID 总和]`；
256 个节点恰好遍历一次时，`sum(range(256))=32640`。这两个字段不是指针。
源表保存 next-node ID，实际读取地址为 `source_base + node_id * stride`，
每次加载前先检查 `node_id < nodes`。`actual=[-1,-1]` 表示输出仍为哨兵，
不能单凭它断言某一跳读到了坏 host 指针。清理阶段 rc=700 也不是注册地址错误的证明。

以下为 smoke 复现命令；已完成本节门禁的机器直接进入 §4，无需重复 smoke。
更新实验 worktree 后，在物理 GPU4/5 空闲时执行，使用新目录保留旧数据：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R8="$LAGOM_REPO/work/h2d-round8"
mkdir -p "$H2D_R8"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_reads.py \
  --nodes 256 --total-reads 256 --grids 1x1 4x32 \
  --samples 2 --epochs 1 --warm-ms 20 \
  --output-dir "$H2D_R8/read-smoke-v2" >"$H2D_R8/read-smoke-v2.log" 2>&1
```

需要退出 0，`success=True correctness=16/16`、`CAPTURE_DEVICE_CHECK 16/16`，
身份为物理 GPU4/5，host 在 node1，
每个 eager/graph/测量校验通过、event 时间均为正。失败则返回 compact 的 ERROR 或日志末尾，
停在 smoke。smoke 性能不用于判读。此工具不请求 profiler，故没有 profile_success 门禁。
若失败，返回
`FAILURE_CONTEXT`、`COMPLETED` 和 traceback 即可，无需额外打印大批指针或上传原始文件。

### 实机 smoke-v2 已通过（2026-10-14，用户简报）

修复提交 `3c0912e1c` 的重试退出码为 0，`success=True correctness=16/16`、
`CAPTURE_DEVICE_CHECK 16/16`。全部 eager、graph、测量输出校验通过，event 时间均正。
物理 GPU4/5 的 UUID、BDF 均匹配；共享 host pool 为 1 MiB，256 页全部 node1，
两条 PCIe 通路均 Gen5 x16。两卡均 P0、MEM 2619，Telemetry errors=0。
这闭合了探针跨卡 capture 缺陷的修复验证，生产 H2D 差距的根因仍未闭合。

本次 smoke 只有两档 grid（1×1、4×32），因此输出是八条 READ 行。
N=128 时每链仅 `256/128=2` 跳，几微秒固定开销不可忽略；源分配也仅 1 MiB。
不能用这些数值宣布单读延迟相同、并行响应相同或排除地址转换因素。
正式运行将总读数增至 32768、源分配增至 512 MiB；N=128 时每链 256 跳。
本地仍仅有 CPU，GPU 门禁依据用户的上述实机简报。

## 4. smoke 通过后只跑一次正式对照

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_reads.py \
  --output-dir "$H2D_R8/read-main" >"$H2D_R8/read-main.log" 2>&1
```

要求退出码 0，`success=True correctness=64/64`、`CAPTURE_DEVICE_CHECK 64/64`，
每配置每源 8 个 ABBA wave、每卡 4 个 wave，
每 wave 3 个计时样本。全部图完成后修改源表，初始化/拷贝/注册/预热都在计时外。
原始 JSON 保存所有样本、代码 SHA、实际 GPU、host 页位置和每波 NVML 状态。
先不做反序、生产修改或 8 卡重测。

短反馈只需：

1. main 的退出码、success/correctness、CAPTURE_DEVICE_CHECK，UUID→GPU4/5、
   512 MiB host pool 在 node1，以及原样的两条 STATE（含 SM_MHz）和 Telemetry errors。
   已通过的 smoke 无需重复或再次贴数值。
2. 从 `read-main/reads-compact.txt` 取两种 stride、host/HBM 的 **1x1、1x16、4x32**
   共 12 条 READ 行，保留 us/hop_ns/Mloads/s/四波均值/SMs；其余配置先留在本机。

```bash
rg '^READ .*grid=(1x1|1x16|4x32) ' "$H2D_R8/read-main/reads-compact.txt"
```

| 观察 | 支持的下一步 |
|---|---|
| N=1 host 已明显慢，HBM 接近 | 低并发读路径成本已有差异；进一步区分地址转换/足迹和传输路径延迟 |
| N=1 接近，增大 N 才拉开，HBM 接近 | 差异与软件并行响应相关；再测生产 kernel 的发射依赖与事务形状 |
| 同 N 的 1-CTA / 4-CTA 不同 | SM 分布影响响应，结合记录的 SM ID；不能归为单纯链数变化 |
| 紧凑 stride 下差距缩小 | 足迹/局部性敏感，下一轮再定向拆解；未直接证明 TLB |
| probe 中全部接近，而生产 SM 基线仍差 | 简化读模式未复现差距，回到 584-byte warp gather 的指令/事务形状 |
| HBM 也出现同方向大差距或状态不匹配 | 先处理 GPU 执行/时钟/测量混杂，不将其专属归因于 host |

以上是待检验假设；不承诺某种结果，不把 static PCIe 差异或一次 sweep 直接写成根因。
