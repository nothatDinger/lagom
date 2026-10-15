# 第十六轮：持续 host gather / bulk CE 的 NVML PCIe Rx 对照

沿用基于 `hash_verify` 的 `perf/hisparse-h2d-diagnosis`。
本轮只回答：**SM有效KV吞吐低时，GPU的PCIe接收流量是否也低？**
使用现有venv、PyTorch和已安装的pynvml。无需新版NCU、Docker、root或新的CUDA内核。

## 1. 为什么测这一项

此前SM带宽是 `有效KV字节数 / copy时间`，不是PCIe物理流量计数。
第十四/十五轮已经发现：D2与CTA8改善HBM执行，却没有host净收益。
继续改变暂存/CTA之前，需要区分“接收通路没有充分利用”与“接收了更多字节，但有效KV占比低”。

NVML `nvmlDeviceGetPcieThroughput(..., NVML_PCIE_UTIL_RX_BYTES)` 提供低频的接收速率近似。
它不是累计字节计数，不提供TLP形状、每笔请求或TLB miss。
脚本原样保留其KB/s读数；另给按1024 B/KB换算的名义GB/s。
CE核验其量级和一致性，不精确鉴定KB的底数，也不把计数器重新缩放成55GB/s。
主要比较使用同卡的 **SM/CE原始Rx比值**，避免单位底数影响这个比值。

## 2. 固定工作与时间口径

同一512MiB（向下取整到完整C4页）NUMA1 pinned allocation，Portable注册flags=1，context=CUDA1。
GPU4/5顺序测量，按ABBA交替。每卡交替CE→SM和SM→CE，正式组各两次。

- SM：原legacy CTA4/1024，B32/miss100，584B/KV，每copy **1,868,800B**。
  随机源、原GPU plan/目的布局、64个bank轮换，逻辑工作集119,603,200B。
- CE：同一host pool中的64MiB连续H2D，通常7个bank轮换。
- 复用现有CopyCase的eager、graph和完整字节校验，SM检查未写区域与页padding。
  actual legacy `.so`摘要照录，不修改kernel、API或服务配置。

每项先采0.5秒idle背景，忽略idle最初150ms和最后50ms；再做原有计算预热和graph预热。
矩阵计算、分配、校验、setup H2D均在测量之外，持续传输期间不执行矩阵计算。

持续传输按约100ms一段执行，每段重放完整的bank graph多次；段结束同步并记录完成字节数。
这是有界提交，不在固定时间内无限向CUDA队列堆积。
外围CUDA Event记录每段GPU时间；原graph内的Event节点保留，因此绝对速率不能直接替代旧copy-only指标。

独立host线程每20ms读NVML Rx/Tx，每200ms读状态；它不调用CUDA、不捕获graph。
正式窗口至少3秒，去掉首尾0.3秒，再选择完全落入区间的完整段。
**逻辑字节数/墙钟时长与NVML平均速率使用同一个稳态区间。**
NVML按实际采样时间加权积分，边界只在相邻样本间插值，不外推缺失区间、不跨越长缺口。
驱动本身的速率平均窗口仍存在，因此结果是近似平均流量；首尾裁剪和稳态检查降低切换污染。

## 3. 自动门禁与失败处理

脚本首先探测两卡的Rx/Tx接口；不支持、权限/绑定错误等异常原样保存，不把错误当零。
CE首先运行以核验计数器。通过条件预先固定：

- 每个传输窗口稳态至少20个样本，时间覆盖≥95%；不跨越超过80ms或4个采样周期的缺口。
- Rx变异系数≤15%；GPU Event区间/同区间墙钟时间在0.90–1.02；每段完整graph提交耗时≤该段GPU Event区间的10%。
  Event区间本身可含等待，所以不能只用Event/墙钟比值断言GPU始终忙；提交耗时界限用于额外限制供给不足。
- idle Rx≤该窗口active Rx的5%。背景不自动扣除。
- CE名义Rx/有效墙钟吞吐在0.85–1.15；CE有效墙钟吞吐≥45GB/s，作为此前约55GB/s参照的质量筛选。

这些是诊断质量门槛，不是统计显著性或物理定律。
零Rx、采样不稳定、CE过慢或量级不匹配都停止本次运行，保留已完成窗口和原始样本。
CE通过不自动证明SM路径的counter覆盖：如果SM接收流量异常接近零、低于合理范围，
仍需区分缓存、流量路径与计数器可见性，不能直接认定PCIe闲置。

CPU单测覆盖非均匀采样、共同时间区间、缺口/错误、错误校准及失败后不输出对照结论。
开发环境没有GPU，NVML与持续graph的实际执行由下面smoke验证。

## 4. 实机执行：先smoke，通过后main

在仓库正常属主账号同步分支，保留已有结果和JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
```

以下命令放在**现有gpuq分配物理GPU4/5的作业内**，从仓库根目录运行。
wrapper设置 `CUDA_VISIBLE_DEVICES=4,5`、NUMA1绑定和日志重定向，不使用GPU0–3或启动服务。

```bash
bash scripts/pd_test/run_h2d_rx.sh smoke "$PWD/work/h2d-round16/rx-smoke"
```

smoke保持正式源/工作量，仅缩短为每项1.5秒、首尾裁0.25秒、一个ABBA epoch。
通过：`success=True counters_available=True correctness=4/4 correctness_valid=True`，
`windows=8/8 quality=True CE_calibration=4/4 results_valid=True`，身份仍为physical4/5，host页在node1。
纯传输段约12秒，另有分配、校验、预热和idle开销；smoke不用于性能/根因定论。

全部通过后直接继续：

```bash
bash scripts/pd_test/run_h2d_rx.sh main "$PWD/work/h2d-round16/rx-main"
```

正式组每项3秒，两个ABBA epoch：每卡4个SM/CE配对，纯传输段约48秒。
通过：correctness仍4/4，`windows=16/16 quality=True CE_calibration=8/8 results_valid=True`。
任何步骤失败就停，不安装工具、不绕过门槛；反馈错误即可。

## 5. 只需要这些反馈

1. smoke的通过状态一行，或失败时的 **rx-compact.txt**。
2. main的 **rx-compact.txt** 全文（约30行，含16个WINDOW、两条PAIRS及STATE）。
3. 如紧凑文件未生成，给run.log最后30行。

`rx.json`含原始采样和每段计数，留在实机，不上传大日志。
git属主失败只记metadata，代码与已加载JIT摘要仍照录，不修改safe.directory。

主要反馈字段：每窗口的有效墙钟/GPU吞吐、原始Rx、名义Rx/Tx、idle背景、样本数/覆盖、Rx变异系数；
每卡四组 `SM/CE_rx` 与 `SM/CE_logical` 比值；以及CE校验和GPU状态。

## 6. 如何用结果决定方向

| 在有效校验下的观察 | 支持的方向 |
|---|---|
| SM Rx接近本卡CE参照，但SM逻辑吞吐明显低 | 额外接收流量/有效字节比例值得优先分析；下一步针对源布局及value/scale访问组织，不继续盲加并发 |
| SM Rx与逻辑吞吐量级接近，且两者均明显低于CE | 本条件下接收速率未到CE参照；继续分析稀疏读取请求的发起、等待及处理效率，不宣布物理带宽上限已到 |
| 介于两者之间、跨波不一致、SM异常零流量或校验失败 | 保留原样，先核对计数器与流量路径，不依据该结果选择CUDA优化 |

即使Rx/逻辑字节比值较大，也不能直接把它解释成某种固定PCIe放大倍数或TLB miss。
NVML是GPU端聚合速率近似，不是总线协议分析器；阶段流量还可能有缓存和控制开销。
本轮是独立copy诊断，不能当作真实decode的链路利用率或服务收益。
