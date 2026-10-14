# 第十四轮：固定分散源，测 D1/D2 的实际收益

沿用 `perf/hisparse-h2d-diagnosis`，基于 `hash_verify`。
本轮只回答：**解除同 warp 跨 KV 的先消费再读取限制，在 B32/miss100 下能省多少时间？**
沿用第十三轮修正版 CUDA 内核，不增加线程、深度、布局或 cache policy 扫描。
使用现有 venv、PyTorch profiler 和 gpuq，无需 Docker、NCU 或安装工具。

## 1. 第十三轮判读结果

作业 `job-20261014-100527-1b05f0a8` 的正确性12/12，每项 eager 和6次动态 graph replay通过。
legacy/D1/D2为34/28/38寄存器，STACK/LOCAL/SHARED均为0。
无spill不等于没有资源代价；D2比D1多10个寄存器，性能结果会包含这项代价。

补充的完整 `codegen-compaction.txt` 确认：

- D2的A source load位于08b0/0900/0920，写barrier 5；B位于0a90/0ad0/0af0，写barrier 2/3/4。
- 两次counts分发SHFL位于0280、05d0，均在A source load之前。
- **0950条件BRA实际有M0x02，不能写成“中间没有任何非零wait”。** 它等待barrier 1，未包含A写入的barrier 5；与首版0770的M0x22不同。
- 配对路径从0950落下继续发B；条件跳转0xd70是单项尾部。
  配对路径的A store到0d00才等待barrier 5，B store分别等待2/3/4。

因此通过“具备测试跨 KV 暂存收益的资格”这一人工判读门。
这不证明运行时一定重叠，也不证明在途PCIe事务数翻倍或物理队列深度为2。
旧 `codegen.json` 中的 `performance_ready=False` 是提取工具的固定字段，无需手改。

新基准先读取实机完整的 `check/iterator.json` 和 `codegen/codegen.json`：
核对12项正确性、源文件/API摘要、codegen输入摘要、上述已审阅的SM90a控制窗口；
JIT加载后再核对三份 `.so` 的摘要。任何不一致都在性能计时前停止。
这个检查只绑定已审阅的v2产物，不是通用SASS依赖分析器；不同codegen需要重新核对。

## 2. 三个比较各自回答什么

| 同卡耗时比 | 用途 |
|---|---|
| D1 / legacy | 遍历重写及其编译变化带来的收益或代价 |
| **D2 / D1** | 同一遍历方案下，两项暂存的净增量，包括额外寄存器和控制成本 |
| D2 / legacy | 候选相对当前实现是否值得进入服务验证 |

主目标固定 **B32、每请求100 miss、584B/KV、每copy 1,868,800B**，同一分散源计划。
三组共享GPU plan、counts和目的buffer；host三组共享同一Portable注册的host allocation；
HBM三组共享相同内容的本卡显存源。全部固定CTA4/1024、legacy scalar load、相同lane分工。
绑定检查会写进结果，避免实际换了plan或buffer却被当成单变量对照。

只保留host/HBM两种源和已有连续CE参照。本轮不测miss1272，也不启动模型服务。
HBM为逻辑载荷吞吐，计时段H2D字节数为0，不能用host时间减HBM时间作为PCIe耗时。

正式组使用512MiB host pool（按完整C4页向下取整）、64个不同的源bank：
轮换的逻辑载荷合计119,603,200B，大于单卡H100 L2容量。
这样避免缩成几个bank后，整个载荷都能装进L2；它不保证所有读取都是cache miss。
不做相邻化或packing，不能与第十一轮相邻布局结果直接当作同一轮配对。

两卡按ABBA执行三轮，共12个设备波次，每卡6波。
**每张卡分别覆盖三种方法的全部六种顺序**，各方法在组内每个位置出现两次；
host/HBM组的先后顺序交替，方法间执行相同计算预热及两次graph预热。
每方法每卡6×64=384次copy；这些graph节点不是384个独立实验，按6个波次报告配对比值。
连续CE放在每组末尾；bulk CE单独采集，作为通路参照，不参与D2/D1因果归因。

无profiler CUDA Event耗时为主要性能指标。之后独立采集一波PyTorch profile，
检查每个方法正确的kernel模板、活动数及event/activity关系；profile只作佐证。
不把profile前后的数字混成同波比较。

## 3. 先做一次配对 smoke

在仓库正常属主账号同步分支，保留原有结果及JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
```

以下命令均放在 **gpuq分配物理GPU4/5的作业内**，从仓库根目录执行。
wrapper设置 `CUDA_VISIBLE_DEVICES=4,5` 和NUMA1绑定、重定向完整日志，使用新的结果目录。
第二个路径参数是本次已经通过的v2检查目录；若实际路径不同，仅替换它。

```bash
bash scripts/pd_test/run_h2d_iterator_perf.sh smoke \
  "$PWD/work/h2d-round13/iterator-check-v2" \
  "$PWD/work/h2d-round14/iterator-smoke"
```

smoke用B2/miss2、70，验证新基准的三组调用、跨request配对和profile归属。
通过条件：

- `success=True profile_success=True results_valid=True`；correctness/graph/profile **30/30**，untouched **24/24**。
- `TIMING_COVERAGE valid=True`、waves **60/60**；BINDINGS **4/4**；`loaded_artifacts_match=True`。
- IDENTITY仍为physical4/5，host VMA实际在node1、Portable flags=1。

smoke仅验证采集，不判性能。通过后直接继续正式组，无需再等一轮回复。
失败则停在这里，反馈紧凑文件或run.log最后30行；不要清cache或绕过产物检查。

## 4. 正式组：只跑B32/miss100

```bash
bash scripts/pd_test/run_h2d_iterator_perf.sh main \
  "$PWD/work/h2d-round13/iterator-check-v2" \
  "$PWD/work/h2d-round14/iterator-main"
```

正式组通过条件：success/profile/results_valid均True；correctness/graph/profile **16/16**；
untouched **12/12**；TIMING_COVERAGE **96/96**；BINDINGS **2/2**。
主目标每方法每卡n=384；bulk 64MiB通常每波7bank（完整C4页向下取整），不可误要求它也64bank。

**只需反馈：smoke一行通过状态，以及正式组 `iterator-perf-compact.txt`。**
文件包含host/HBM两卡的三组均值、三个同卡比值及6波配对比值、profile活动/gap、CE参考、状态。
`pair.json`、`pair.trace.json`、run.log保留本地，不需要上传。
git属主查询失败仍只记metadata错误；代码和已加载JIT摘要照常核对。

## 5. 如何决定下一步

优先看慢卡GPU5主目标的**无profiler D2/D1和D2/legacy**，同时看GPU4与各波方向。
以下是下一轮投入的筛选规则，不是统计显著性声明：

- 两比值均稳定低于1，例如节省≥5%、至少5/6波同向，且状态无明显分化：
  支持“现有读取组织限制了该载荷的性能”，再考虑服务验证；不要求卡间比值变为1。
- D1明显更快而D2/D1接近1：收益主要落在遍历/编译变化，优先考虑D1，不能声称两项暂存有效。
- D2/D1更快但D2/legacy不快：暂存可能补回了重写成本，当前候选没有净收益。
- D2/D1在约±2%内或更慢：降低此D2候选优先级，不继续自动展开深度扫描。
- 中间地带、波次反复换向或明显状态漂移：保留不确定性，先看已有逐波数据，不预设必须加跑。

HBM若同样改变，只能说明收益包含通用执行因素；host更强的收益支持host路径更敏感，
不能把两个比值或时间简单相减解释为纯PCIe部分。
即使D2获益，也不能据此确定ATS/TLB、PCIe tag或硬件队列是根因。
服务吞吐/TPOT仍需后续实测，本轮不提前宣布生产优化成功。

本地只完成CPU侧单元测试、回归检查和脚本检查；新配对入口的CUDA执行与profile匹配由§3实机smoke验证。
