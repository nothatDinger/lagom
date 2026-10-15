# 第十六轮：持续 host gather / bulk CE 的 NVML PCIe Rx 对照

沿用基于 `hash_verify` 的 `perf/hisparse-h2d-diagnosis`。
原目标：区分“SM有效KV吞吐低且PCIe接收速率也低”和“接收了更多字节但有效KV比例低”。
**GPU4与GPU5的采集自检均已完成；本轮不再请求GPU作业或补充日志。**
下方命令保留用于复现，不续跑旧smoke/main。使用现有venv和pynvml，无需NCU、Docker或root。

## 1. 首次smoke实际发现了什么

- 身份/NUMA/字节正确性通过。GPU4 CE稳态有效吞吐55.06 GB/s、NVML名义Rx 60.53 GB/s，通过既定粗校验。
  两者相差约9.9%，不能把名义Rx当精确的有效PCIe载荷。
- SM pilot在持续采样线程启动前，单copy已经约115.46us；该pilot的host跨度包含同步等待，
  host跨度与GPU区间接近**不证明graph.replay调用本身没有阻塞**。
- SM持续段固定14次bank graph/段，前10段约105.6ms，后段下降到60.5ms；未达到稳态。
- 每次Rx+Tx查询耗时约41ms，所以设定20ms实际得到约41ms间隔。
- 旧SM窗口Rx CV=0.15082略超0.15，CPU提交/GPU区间最高0.685又超0.10，两个门槛分别失败。
  0.685不等于GPU空闲68.5%；提交时间可与GPU执行重叠，异步API也可能阻塞。

因此旧SM窗口不能支持H2D根因结论，也不能仅凭这些信息把问题归咎NVML。
下面先消除反复提交带Event graph的干扰机会，再用相同graph检验采样开关的影响。

## 2. 具体改动与固定条件

继续暴露物理GPU4/5，Portable注册context仍为CUDA1；每次**只在一张选定卡执行搬运**。
`--check-device 0`选择GPU4（默认），`--check-device 1`选择GPU5。
另一张卡只保留身份映射/注册context的既有用途，不构造其copy case或执行计算预热。

- 同一512MiB NUMA1 pinned pool；SM仍legacy CTA4/1024、B32/miss100、584B/KV、64个随机bank。
- CE仍是同一host pool中的64MiB连续H2D、7个bank。CUDA kernel、load policy、plan、目的地址不变。
- 每个方法先capture一个bank序列，取5次外围Event时间的中位数估算约100ms的工作量。
  再capture包含若干完整bank序列的chunk graph，**每段只调用一次graph.replay**；段结束同步。
  graph内部没有逐copy计时Event；仅在graph外记录一对Event。图大小在四个开关阶段保持一致。
- 新graph另做reset→replay→同步→全字节校验；SM同时检查未写区域和padding。
  `per_copy_timing_events=0`和`launches_per_chunk=1`是脚本构造约束，不冒充profiler节点统计。
- 每阶段先运行**同一copy负载**2秒，再测量2.5秒。不使用matmul预热。
- 每方法执行**OFF→ON→ON→OFF**四阶段：OFF不创建采样线程；ON仅每50ms查询Rx，不查Tx。
  两种阶段都只在测量前后查询状态，持续区间内不插入状态查询。
- ON前采0.7秒idle背景。查询耗时与实际时间戳照录，Tx未采集记None，不伪造为0。

CE的四阶段完成后再做SM四阶段，总计8个窗口；准备/预热之外的测量约20秒，copy预热约16秒。
这是采集方式自检，既不归因旧smoke故障的唯一来源，也不是新CUDA优化或生产吞吐结果。

## 3. 预先固定的质量检查

所有窗口去掉首尾0.3秒，只保留完整传输段，逻辑字节数与Rx积分使用同一个墙钟区间。

- GPU Event区间/墙钟区间0.90–1.02；每段单次提交耗时/GPU区间≤0.10。
- 段耗时按bank序列数归一化，CV≤0.10；最后三分之一/最初三分之一耗时比0.90–1.10。
  原smoke那种两个明显平台不能仅凭GPU/wall≈1过关。
- ON的Rx至少20个样本、覆盖≥95%，不跨越超过80ms或4个采样周期的缺口；Rx CV≤0.15。
- idle Rx≤active的5%。CE名义Rx/逻辑墙钟吞吐0.85–1.15，CE有效吞吐≥45 GB/s。
- 同方法两个OFF的耗时比0.90–1.10；两个ON分别相对OFF平均耗时的比值都在0.90–1.10。
  每个方法必须四阶段齐全且顺序正确，不用某一次较好的ON替代全部结果。

上述10%是排除大幅扰动/漂移的筛选条件，不是零干扰证明或统计显著性检验。
两个ON的实际比值都输出，即使合格仍需审查方向与量级。
NVML是低频近似速率，保留原始KB/s，另按1024 B/KB给名义值；不重缩放、不把它变成TLP/TLB计数。

某个窗口质量不合格，仍完成本方法余下开关阶段，以暴露前后漂移；正确性、CUDA或API异常立即退出。
`success=True`只代表8个阶段执行完成，最终质量取决于`results_valid`；失败时保留全部已获得数据。
两个ON的query异常也保存在原始采样中，不能按0或成功处理。
端点状态不证明区间内恒频，采样开关对照也不证明SM与CE计数器覆盖完全相同。

## 4. 已完成的GPU4/GPU5自检复现方法

GPU4的v2结果：全部门槛通过，两个ON/OFF耗时比均接近1，OFF之间无明显漂移。
SM有效吞吐约30.87 GB/s，CE约55.36 GB/s。
原始Rx读数SM约38,682,593 KB/s，CE约59,890,533 KB/s；两者之比约0.6459，
有效吞吐之比约0.5576。按1024 B/KB换算的名义Rx为SM约39.61 GB/s、CE约60.97–61.69 GB/s。
这些是NVML近似读数及其比例，不能直接当作物理链路利用率或PCIe事务放大。
查询耗时约20.5ms，不是采样周期；50ms的计划周期对应约20Hz，并非48Hz。

GPU5使用相同check选择CUDA1/物理GPU5，其他工作量、采样与注册context保持不变。

从仓库正常属主账号同步，保留原始rx-smoke与JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

在现有gpuq分配物理GPU4/5的作业内，从仓库根目录执行：

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_rx.sh check "$PWD/work/h2d-round16/rx-check-gpu5" 1
```

wrapper设置`CUDA_VISIBLE_DEVICES=4,5`和NUMA1绑定，重定向日志，不覆盖旧目录。
本地CPU检查不替代实机CUDA capture/字节校验，本次check包含这些验证。

期望：

```text
success=True counters_available=True correctness=2/2 correctness_valid=True
chunk_graphs_valid=True windows=8/8 CE_calibration=2/2 results_valid=True
```

两条CONTROL都应为`{'valid': True, ...}`。
GRAPH的device/capture_device以及STATE应为d1，IDENTITY d1应为physical5/B8:00.0。
即使全部通过，本次也到此结束，**不自动续跑旧main**。wrapper暂时拒绝旧smoke/main参数。

## 5. 只需要反馈一份紧凑文件

反馈`work/h2d-round16/rx-check-gpu5/rx-compact.txt`全文（约20行）。
它包含：两条GRAPH、8条WINDOW、两条CONTROL、身份/NUMA/JIT以及端点STATE。
重点是两个ON/OFF耗时比、两个OFF的漂移、提交占比、段耗时CV、Rx查询耗时与CE校验。
若紧凑文件未生成，给run.log最后30行。rx.json留在实机，不上传大日志。

若ON稳定慢于OFF，就先解决探针扰动；若OFF自身漂移，先解决预热/执行状态；
只有相同graph的开关对照、稳态与CE校验都合格后，才恢复原本的有效字节/Rx对照目标。

## 6. 两卡完成后的有效结论

两份报告均correctness=2/2、windows=8/8、CE_calibration=2/2、results_valid=True。
以下取各自两个ON窗口的均值；两次为独立作业、独立allocation，同协议/布局/NUMA，
不能写成这两次直接共享了一个host allocation。

| 指标 | GPU4 | GPU5 |
|---|---:|---:|
| CE有效墙钟GB/s | 55.3565 | 55.1998 |
| SM有效墙钟GB/s | 30.8688 | 22.88865 |
| SM原始Rx KB/s | 38,682,593.25 | 28,778,806.95 |
| SM名义Rx GB/s（1024 B/KB） | 39.61098 | 29.46950 |
| SM名义Rx/有效吞吐 | 1.28320 | 1.28752 |

GPU5/GPU4的SM有效吞吐比为0.74148，SM原始Rx比为0.74397；
单位有效字节对应的Rx比值在两卡之间仅相差约0.34%，未显示出与35%耗时差同量级的额外接收字节。
因此不支持“GPU5因每KV接收更多字节才慢”的解释；观察到的是Rx与有效吞吐近似同比下降。
这仍不是对PCIe事务数、请求大小、TX开销、链路总利用率或硬件队列的直接测量。

两个ON/OFF均近1，两个OFF无明显漂移；本条件下未检出显著的采样扰动。
GPU5 CE trial2的查询耗时最大83ms、Rx CV约0.133，保留该窗口，不择优删除。
SM的CV较低；卡间单位字节比较直接使用SM读数，不依赖用该CE窗口重新缩放计数器。

目前没有证据支持“SM已达到与bulk CE相同的Rx速率，只因有效字节比例低才显得慢”。
但NVML口径不足以宣布物理链路未饱和，也不能直接把1.28解释成固定PCIe放大倍数。
结合此前D2和CTA8的负结果，不能继续仅凭“在途不足”盲加并发；
尚待区分连续SM读取本身的实现/路径限制与稀疏gather的组织开销。
下一步实现见[第十七轮连续SM对照](README_h2d_round17.md)。
