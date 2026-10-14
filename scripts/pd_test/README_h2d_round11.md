# 第十一轮：只检验源地址布局对当前 SM 搬运吞吐的影响

分支 `perf/hisparse-h2d-diagnosis`，继续基于 `hash_verify`。
物理 GPU4/5、NUMA1、已有原生 venv；使用 CUDA Event 和现有 PyTorch profiler，不需要 NCU。

## 1. 本轮唯一问题

相同 KV 内容、miss 数、目的地址、legacy kernel 和 CTA4/1024 下，
将随机分散的源槽位换为相邻 C4 槽位，搬运耗时是否明显下降？

这不是第七轮排序：排序保留了分散地址集合；本轮在计时前重新放置 KV 内容，
使源槽位相邻。也不是生产 packing 优化：准备成本不在计时内，先测布局本身的影响。
没有修改服务默认、planner、CUDA kernel、load policy 或流水深度。

## 2. 对照怎样保证正确和可比

`bench_h2d_pair.py --source-layout-control --device-source-control` 执行五种方法：

| 方法 | 源存储 | 源槽位 | 作用 |
|---|---|---|---|
| `sm` | host | 分散 | 当前真实搬运 kernel 的新进程基线 |
| `sm_adjacent` | 同一 host allocation | 相邻 | 本轮唯一干预 |
| `sm_hbm` | 本卡 HBM | 同一分散 plan | 辅助判断 GPU 内存路径对布局的响应 |
| `sm_hbm_adjacent` | 同一本卡 HBM allocation | 同一相邻 plan | 对应的 HBM 控制 |
| `ce` | 同一 host allocation | 连续 byte 区间 | 等载荷的连续 CE 参照，另有 bulk CE |

准备阶段在同一 host pool 的前部预留**所有载荷组、所有 bank**的相邻区域，
分散 plan 从剩余区域随机取唯一槽位。两者不重叠；较晚载荷的 packing 也不会覆盖较早载荷。
分别拷贝每条 KV 的 576 B value 和 8 B scale，遵守原 64-token/page 的 C4 布局：
`[64×value][64×scale][padding]`，每页 37440 B。不能用 `584×slot` 当物理地址。

每组记录所有 bank 的 CPU 字节检查、原始源保留及页足迹。四个 SM 组在每张 GPU 上
共享实际 GPU miss counts、real batch、目的 plan 和目的 allocation；运行时检查指针，
源 plan 的形状/stride 和 launch/load policy 也必须一致。HBM shadow 在全部 packing 完成后创建。
所有 bank 都做 eager 字节检查，graph replay 另校验最后一 bank；value、scale、未写区域及
padding 的 GPU 输出始终按**原始分散源**校验，避免只验证自洽的新 plan。

每波使用不同源 bank；每次图执行前统一计算预热和图预热，计时只包含原搬运 kernel。
GPU 顺序为 ABBA；每个设备的 wave 内交替测分散/相邻，下一 wave 反转顺序。
两卡各自先测分散、先测相邻的次数相同。`--reverse-method-order` 可以以后用于追查漂移，
本轮先只执行一次正式对照，不预先增加反序、线程扫描或服务测试。

记录的 `virtual_4K_pages` 是 value/scale 涉及的虚拟 4KiB 页数，不是实际物理页、
IOMMU 页、TLB miss 或 PCIe TLP 数。相邻组访问足迹较小是干预的一部分；
预热、缓存复用和地址局部性一起影响结果，不把收益单独归因到其中某一级。

## 3. 同步代码并运行短 smoke

在正常属主账号中同步；若有未提交修改，先保存，避免覆盖实机本地工作。

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
```

通过已有 gpuq 方式分配**物理 GPU4/5**，下面的命令在作业内执行。
两个 GPU 需无其他负载；不启动模型服务。包装脚本自动绑定 NUMA1、设置可见设备并保留日志。
使用全新目录，重复运行要更换目录名；不清 JIT cache。

```bash
bash scripts/pd_test/run_h2d_layout.sh smoke "$PWD/work/h2d-round11/layout-smoke"
```

smoke 为 B2、miss2/70、4 MiB pool、4 banks，用于测试页边界、packing、两设备 graph 和解析；
smoke 的带宽不作性能判断。脚本输出 `layout-compact.txt`，完整报告/trace 留在目录内。

通过条件：

- 作业/脚本退出 0，`success=True profile_success=True ready_for_main=True`。
- correctness/graph **22/22**，untouched **16/16**；`PROFILE_COVERAGE validated=22`、errors=[]。
- 两条 `LAYOUT ... prepared=True banks=4`，`LAYOUT_BINDINGS checked=4/4`，
  `TIMING_COVERAGE valid=True waves=44/44`。
- `IDENTITY` 对应物理 4/5（98:00.0 / b8:00.0）；HOST 的 mapped_bytes=bytes，
  对应 VMA 的页面全部 node1、Portable=1。按完整 C4 页截断后，mmap 会向 OS 页大小圆整，
  因此 `buffer_covers_whole_vma=False` 本身正常；不按比例拆分整 VMA 的 NUMA 页计数。
- 相邻组 footprint 的 C4 页数是预期的小范围；两组内容与原始输出 oracle 校验已通过。

任何一项失败：停在 smoke，返回 `layout-compact.txt` 或 `run.log` 尾部约 30 行，
保留 `pair.json` 和原始 trace，不需要上传它们。当前开发环境只有 CPU，
实机 eager/graph、CUPTI 和硬件身份仍需这一步验证；本地 CPU 通过不能替代。
本地已通过 117 项测试，覆盖 C4 value/scale/padding、跨页及多 bank packing、
原始源 oracle、旧 paired benchmark 兼容性、graph 计时解析和报告覆盖检查；
另通过 Python 静态检查、CLI 参数加载和 shell 语法检查。

## 4. smoke 全部通过后，只运行一次正式对照

无需等待额外确认，全部通过即可在已有 GPU allocation 中运行：

```bash
bash scripts/pd_test/run_h2d_layout.sh main "$PWD/work/h2d-round11/layout-main"
```

正式配置为 B32、miss100/1272，即 **1,868,800 / 23,771,136 B**；
512 MiB host allocation（按完整 C4 页截断少量尾部）、8 个源 bank、2 ABBA epochs。
每卡每种方法有 4 个无 profiler wave，每 wave 8 个 copy，即 **32 个计时样本**。
smoke 和 main 的方法/正确性/profile 项数都是22，main 的 `TIMING_COVERAGE` 为88/88。
测量阶段为 `compute`；独立 profiler 阶段为 `profile_compute`，二者严格分开。
整个 Python 进程完成 profiler 导出后脚本才退出，不使用服务端的 profile 启停驱动。

相邻预留区域减少了分散组可选的地址范围；main 的分散源仍覆盖约三百 MiB。
实际区域与每 bank 的足迹保留在 `pair.json` 的 `layout_plans`，不能直接沿用旧基准的数值。
必须确认本次 `sm` 仍显示原主要现象，CE 仍提供合理参照；如没有复现，先返回本次结果，
不将本次差异用于宣称解释了旧服务的约23 GB/s。

## 5. 只需回传一份紧凑结果

全部正常时：smoke 用一行确认门禁；正式运行返回 `layout-main/layout-compact.txt`（约四十行）。
重点字段是：

1. success/ready、identity、HOST、LAYOUT、FOOTPRINT、LAYOUT_BINDINGS 和完整计时/profile覆盖。
2. `RESULT compute ... host`：两卡、两档的分散/相邻 event µs、有效载荷 GB/s、
   adjacent/random 及四个 wave 比值；这是性能主口径。
3. 对应 `RESULT ... hbm` 与 `CE`；HBM 的每次 H2D bytes 为0，其 GB/s只是逻辑处理率。
4. `PROFILE_RESULT`：同 profiler 段中的 kernel activity 与 event−activity gap，
   用于辅助确认变化发生在 kernel 内，不替代无 profiler 性能结果；STATE 给出波前后状态。

`layout_prepare_us` 包含 plan 生成、packing、CPU 校验和 footprint 统计，**不是生产 packing 成本**。
完整 code hashes 及 git 失败原因保留在 `pair.json`；gpuq 异属主导致 git=None 不使报告中断。

## 6. 本轮结果如何决定下一步

| 观察 | 可以作出的判断 |
|---|---|
| 基线有效，相邻 host 显著更快，四波方向一致 | 源布局及其局部性成本确实限制当前搬运；量化节省的 kernel 时间 |
| 相邻 host 接近同载荷 CE 参照 | 当前 legacy kernel/小 grid 在更有利源布局下可以接近该参照，不能视23 GB/s为固定代码上限 |
| 相邻只改善几个百分点或无一致改善 | 不把源分散作为当前主要原因，不继续扩展生产 packing |
| HBM 也明显改善 | 地址局部性/读取组织在 GPU 内存路径也有影响；不能把 host 差值直接当作PCIe耗时 |
| CPU/GPU/graph/profile 校验失败或基线没有复现 | 保留证据并先处理可比性，不输出根因结论 |

无论结果如何，本轮不从 footprint 推算事务数，也不要求两卡比值必须变成1。
只有对目标 host 搬运的实测时间贡献决定布局是否值得继续研究。
