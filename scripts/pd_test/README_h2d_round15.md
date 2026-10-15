# 第十五轮：legacy CTA4/CTA8 的最小配对对照

沿用基于 `hash_verify` 的 `perf/hisparse-h2d-diagnosis`。
本轮只检验：**默认四个CTA的配置是否限制当前B32/miss100的host搬运吞吐。**
使用已验证的legacy kernel、现有venv/PyTorch profiler和gpuq；无需安装工具或Docker。

正式反馈：host CTA8/4为1.0296/1.0126，无净收益；HBM为0.7336/0.7047，六波均快。
该结果不排除全部并行度问题。后续先测有效载荷与接收流量的区别，见[第十六轮](README_h2d_round16.md)。

## 1. CTA分工的源码核对

当前服务路径是 `hisparse_coordinator.py::_run_swap_in_verify_window`：
先用 `plan_cache_window_hash_mla` 生成miss plan/counts，再启动 `copy_cache_planned_mla`。
`_prefetch_copy_blocks=4` 控制第二步的copy grid，不控制前面的hash planner grid。
本轮重放同一份既定plan，只测copy。

`hisparse.cuh::copy_cache_planned_kernel`（894行起）没有统计专用CTA与搬运专用CTA：

```cpp
warp_global = blockIdx.x * 32 + threadIdx.x / 32;
total_warps = gridDim.x * 32;
for (each real request) {
  cnt = __shfl_sync(...);  // 每个warp都读取/分发counts并推进start
  m0 = positive_mod(warp_global - start, total_warps);
  for (m = m0; m < cnt; m += total_warps)
    copy_miss_item(...);  // 属于本warp的项才搬运
  start += cnt;
}
```

扁平KV索引k由warp `k % total_warps`负责。同一warp可能在某request没有项，
但仍遍历该request；不是固定指定某些CTA只统计。
`num_real_reqs`屏蔽padding请求；`SkipIO`若启用会对所有CTA略过搬运。本轮明确legacy且SkipIO=false。
搬运内部的lane分工（value前缀、tail/scale）也不由CTA编号选择。

均匀B32/miss100的分工如下，compact报告自动输出 `SCHEDULE`：

| copy配置 | 总warp | 每warp KV | 每CTA KV | 全部KV | warp级request遍历次数 |
|---|---:|---:|---|---:|---:|
| CTA4×1024 | 128 | 25 | 800、800、800、800 | 3200 | 4096 |
| CTA8×1024 | 256 | 12–13 | 前四个416，后四个384 | 3200 | 8192 |

八个CTA全部参与搬运。单看第一个request会误判：它只有100项，CTA4–7当时不搬，
但后续request的start移动后会轮到它们。
表中是逻辑工作量，不是GPU活动计数、DRAM读取次数或PCIe事务数。
更多warp会重复更多request判断和索引计算；因此本实验测试的是增加grid的净效果，
不是把所有性能变化都归因于某一个硬件队列。实际SM放置也不能由CTA数量直接推出。

源码的小grid注释明确写了降低SM占用、与side-stream计算重叠的意图。
该意图不能证明四个CTA在DSPARK的新负载下最优，也不能证明单KV暂存深度是同样的设计权衡。

## 2. 固定项、计时与校验

主目标固定B32、每request100 miss、584B/KV，**每次copy始终1,868,800B**。
两组共享GPU src/dst plan、counts、real和目的buffer。
host两组共享同一Portable注册的NUMA1 allocation；HBM两组共享本卡同内容显存源。
固定legacy、1024线程/CTA、stride3072、随机目的槽位以及分散源地址。

仅改变调用的runtime `num_blocks=4|8`；CUDA内核/API源码没有改动，JIT模板参数也不含num_blocks。
实际加载的单一legacy `.so` 及摘要会记录；无需重新提取SASS。
source为512MiB向下取整到完整C4页，64个不重叠bank轮换，逻辑载荷合计119,603,200B。
这降低全部数据装进L2的风险，不保证每条读取都是cache miss。

保留host/HBM与连续CE参照；本轮不加入D1/D2或其他grid扫描。
两卡按ABBA执行三轮，每卡6波。每卡4→8和8→4各三次，host/HBM先测也各三次。
每个方法前执行相同计算预热、两次graph预热，计时里不包含矩阵计算。
六波对应384个graph copy节点；节点不是384个独立实验，报告按波配对。

无profiler CUDA Event为主要指标。随后独立采集profile，严格检查：
每个scope内copy数、实际device、每条SM activity的grid=(4或8,1,1)、block=(1024,1,1)。
读取不到grid/block或不匹配即校验失败，不凭method名称推断捕获成功。
profile activity和它同波的event用于佐证，不能与无profiler波交叉相减。
HBM是逻辑吞吐，计时段H2D字节数为0，host/HBM时间不能相减作为PCIe耗时。

## 3. 同步后先执行smoke

在正常属主账号中同步分支，保留已有结果和JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
```

在gpuq分配物理GPU4/5的作业内、仓库根目录执行以下命令。
wrapper固定可见设备4/5和NUMA1绑定，并将完整输出保存在run.log。
不启动模型或使用GPU0–3。

```bash
bash scripts/pd_test/run_h2d_grid.sh smoke "$PWD/work/h2d-round15/grid-smoke"
```

smoke用B4/miss2、70；后者有280条KV，足以让CTA8的全部256个warp都分到搬运项。
smoke仅验证采集，不判性能。通过条件：

- `success=True profile_success=True results_valid=True`。
- correctness/graph/profile **22/22**，untouched **16/16**。
- TIMING_COVERAGE **44/44**；BINDINGS **4/4**；GRID_ACTIVITY **16/16**；single_legacy_artifact=True。
- IDENTITY仍为physical4/5；host在node1，register_flags=1。

若失败停在smoke，给grid-compact.txt或run.log末尾30行，不绕过grid校验。
如果仅profile维度字段缺失，错误行会保留实际观察值；原始trace留本地，后续可定向提取。

## 4. smoke通过后直接执行正式组

```bash
bash scripts/pd_test/run_h2d_grid.sh main "$PWD/work/h2d-round15/grid-main"
```

正式门禁：success/profile/results_valid均True；correctness/graph/profile **12/12**；
untouched **8/8**；TIMING_COVERAGE **72/72**；BINDINGS **2/2**；GRID_ACTIVITY **8/8**。
主目标各方法每卡n=384。bulk 64MiB每波通常7bank，由完整C4页取整决定。

只需反馈smoke通过一行，以及正式组 **grid-compact.txt**。
它含CTA4/8的host/HBM时间与带宽、六波配对比值、实际grid校验、profile活动/gap、CE和状态。
pair.json、pair.trace.json、run.log留在实机，不需要上传。
git属主失败只记metadata错误，代码和实际JIT摘要照录，不修改safe.directory。

## 5. 按结果决定后续投入

`8/4`是同卡耗时比，小于1才是加速。优先看GPU5的无profilerhost结果，核对GPU4与逐波方向。
以下是投入筛选条件，不是统计显著性声明：

| 观察 | 解释和下一步 |
|---|---|
| host稳定节省例如≥5%、至少5/6波同向，状态无明显分化 | 当前grid限制净吞吐；再准备真实decode验证，衡量更多SM对计算的影响 |
| host约±2%内、HBM改善 | 通用执行受益没有转成host收益，降低“增加供给即可解决”的优先级 |
| host不改善、HBM也变慢 | 重复遍历/调度成本可能遮住收益；不能宣布host已饱和，先分析现有活动与执行路径 |
| 波次换向、状态漂移或中间幅度 | 保留不确定性，先看现有配对波，不自动扩大扫描 |

单次grid干预不能确定SM请求队列、TLB、PCIe tag等具体根因；即便micro加速也需服务实测。
开发环境只验证CPU分析/解析和脚本；实际CTA8 graph与profile维度由本轮实机smoke确认。
