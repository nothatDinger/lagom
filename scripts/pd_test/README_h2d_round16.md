# 第十六轮：持续 host gather / bulk CE 的 NVML PCIe Rx 对照

沿用基于 `hash_verify` 的 `perf/hisparse-h2d-diagnosis`。
原目标：区分“SM有效KV吞吐低且PCIe接收速率也低”和“接收了更多字节但有效KV比例低”。
**当前只执行下面的采集自检，不续跑旧smoke/main。** 使用现有venv和pynvml，无需NCU、Docker或root。

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

继续暴露物理GPU4/5，Portable注册context仍为CUDA1；本次**只有GPU4执行搬运**。
GPU5只保留身份映射/注册context的既有用途，不构造其copy case或执行计算预热。

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

## 4. 只执行一次check

从仓库正常属主账号同步，保留原始rx-smoke与JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

在现有gpuq分配物理GPU4/5的作业内，从仓库根目录执行：

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_rx.sh check "$PWD/work/h2d-round16/rx-check-v2"
```

wrapper设置`CUDA_VISIBLE_DEVICES=4,5`和NUMA1绑定，重定向日志，不覆盖旧目录。
本地CPU检查不替代实机CUDA capture/字节校验，本次check包含这些验证。

期望：

```text
success=True counters_available=True correctness=2/2 correctness_valid=True
chunk_graphs_valid=True windows=8/8 CE_calibration=2/2 results_valid=True
```

两条CONTROL都应为`{'valid': True, ...}`。
即使全部通过，本次也到此结束，**不自动续跑旧main**。wrapper暂时拒绝旧smoke/main参数。

## 5. 只需要反馈一份紧凑文件

反馈`work/h2d-round16/rx-check-v2/rx-compact.txt`全文（约20行）。
它包含：两条GRAPH、8条WINDOW、两条CONTROL、身份/NUMA/JIT以及端点STATE。
重点是两个ON/OFF耗时比、两个OFF的漂移、提交占比、段耗时CV、Rx查询耗时与CE校验。
若紧凑文件未生成，给run.log最后30行。rx.json留在实机，不上传大日志。

若ON稳定慢于OFF，就先解决探针扰动；若OFF自身漂移，先解决预热/执行状态；
只有相同graph的开关对照、稳态与CE校验都合格后，才恢复原本的有效字节/Rx对照目标。
