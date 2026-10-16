# GPU4：SM gather 与连续 CE 的并发干扰

只回答：SM 的吞吐平台是否表现为与 CE 共享的受限服务能力？不改生产 kernel，
不安装 NCU、不读 Rx、不改时钟、不用 GPU5。本实验不能直接识别硬件锁或队列。

## 设计

沿用双 reader harness，A=legacy SM CTA4/1024、B32/miss100；B=连续 H2D CE。
同一 NUMA1 Portable 注册 512MiB host allocation，SM 源限前半，CE 用后半
独立16MiB范围，两者目的 buffer 独立。SM plan/地址分散方式与上一轮相同。
CE 同一16MiB区域重复传输，计数按每次实际复制字节累加。

SM smoke 4 banks / main 64 banks；预热后 pilot 各5次取中位数，选择 CE 重复
次数，使单路图时长接近。重复次数固定后才采集正式数据，不动态改变工作量。
main 的单路 CE/SM 中位时长比要求0.8–1.25；smoke短图不要求此门。
CE 单路每波应≥45GB/s，否则不解释混合结果。

六组：A、B、serial_AB、serial_BA、parallel_AB、parallel_BA；main六波交错顺序。
每段只提交每路一个图；外层CUDA事件统计合计时间，每路外部事件统计自身时长。
总字节必须分别相加，SM字节与CE字节不同，不能套用双SM的乘2计数。
每路GB/s=本路字节/本路持续时间；RETENTION为同trial并发/单跑的本路吞吐比。
这是各路完整生命周期的比值，不是仅重叠区间内的瞬时吞吐。

性能测量结束后独立采集三份trace。SM要求64条（smoke4）、grid4/block1024；
CE要求与选定重复数一致，HtoD、每条16MiB；设备CUDA0，串行同stream、并发不同stream。
报告两路各自activity的重叠比例、交集时长；启动偏斜也保留。
不得用profiler时长与无profiler性能拼成同一次测量。

## 执行

同步分支后，在gpuq物理GPU4 allocation内：

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  smoke "$PWD/work/h2d-sm-ce/smoke-v1" sm-ce
```

smoke门禁：success/results_valid=True、checks=8/8、profiles=3/3、
CONFIG partner=ce、labels A=SM B=CE、CE_calibration=True；
绑定source_disjoint/independent_outputs为True，身份GPU4/NUMA1正确。
重叠率和吞吐保持率是观测值，不作为smoke成败条件。
全部满足后直接续跑：

```bash
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  main "$PWD/work/h2d-sm-ce/main-v1" sm-ce
```

反馈smoke门禁一行以及main/result/streams-compact.txt全文。
streams.json及三份trace留本地。任何一步失败停止，反馈compact和run.log最后30行。
不自行降低门槛，不自动追加其他配置。

## 判读

先检查每路重叠比例、启动偏斜、单跑时长匹配、逐波状态；若并发时两路时长明显
不对称，尾部独跑会稀释RETENTION，不能直接当成共同稳态的吞吐。
必要时再用已有trace检查交集与尾段，不先扩大实验。

SM速度基本保留且CE提供额外吞吐，支持SM特有路径限制，而非共同通路无余量。
两路明显互相拖慢，支持共享服务/仲裁因素，但不能据此确定PCIe物理带宽饱和。
影响不对称可提示优先级或不同瓶颈，不能单凭一轮断言硬件公平策略。
合计逻辑吞吐不是PCIe线速；不把SM+CE逻辑GB/s与55GB/s机械相减归因。

本地仅CPU解析测试/语法检查，CUDA copy graph及实机trace字段仍须smoke确认。
