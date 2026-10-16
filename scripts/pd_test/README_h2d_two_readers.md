# 两个独立 CTA4 reader：执行串行化与共享吞吐对照

目标：观察两个独立 host gather 任务能否在 GPU 上重叠，以及合计吞吐是否增加。
只用物理 GPU4，A/B 各 CTA4/1024，B32/miss100，legacy 原生产 kernel；不改服务默认值。
不安装 NCU，不改时钟/驱动，不读取 Rx。该实验不能直接检测硬件内部锁或内存请求仲裁。

## 固定条件

- 同一512MiB Portable 注册 allocation，NUMA1，按完整C4页划成两块互不重叠源区域。
- A/B 各自独立 src/dst plan、counts、real、目的 buffer和非默认stream。
- 每次copy各1,868,800B；smoke各4个bank，main各64个bank，每bank源KV唯一。
- 注意：每reader源范围约256MiB，不等于旧实验的512MiB范围；以当轮A/B单路作基线。
- 每个reader一个已捕获graph，内部只在整个bank序列前后记录external CUDA Events。
- 计时前250ms计算预热及两次同模式graph预热；所有性能数据无profiler。

六配置：A单独、B单独、serial_AB、serial_BA、parallel_AB、parallel_BA。
main六波，奇偶波反转配置顺序；smoke两波，仅验证执行与报告，不作性能结论。
并发组由协调stream记录共同开始event，两个worker stream各wait该event后replay；
两次replay均已提交后，协调stream才wait两端完成event。最后记录统一结束event。
无CPU同步插入两次并发提交之间，无GPU自旋屏障。记录两graph实际event起点偏斜。
串行组在同一协调stream依次replay同一A/B graph。

计数口径：单路总字节=banks×1,868,800；双路总字节再乘2。
主指标统一开始至全部结束的CUDA Event总时间、合计有效吞吐。
同时保留每reader graph耗时、起止位置、CPU墙钟及前后NVML状态。
不要用并发两倍工作量的总时间，直接与旧单kernel CTA8比较。

## 正确性与时间线

A/B先各做eager首末bank、graph最后bank全字节与padding检查；
随后六种调度方式全部检查目的字节。实际JIT从进程映射提取SHA。
性能测量后另采serial_AB、parallel_AB、parallel_BA三份PyTorch profiler trace。
每份恰含2×banks个copy activity，逐项检查GPU0、grid4、block1024。
parallel必须有两个stream且各banks个activity；缺失/异常保留报告并失败，不能解释为硬件锁。
串行参照用于检查采集和依赖是否符合预期。

重叠量计算GPU kernel activity区间的交集，不使用两个graph外包络重叠冒充kernel重叠。
profile只看执行重叠，性能用无profiler数据。低重叠也不能自动认定锁；先查启动偏斜、
CUDA依赖和调度。明显重叠但总吞吐未增长支持共享处理能力受限，不能排除内部请求串行服务。
功率限制标志、频率等端点数据不证明波内状态恒定，保留逐波数据供判读。

## 执行

属主先正常同步分支；在gpuq物理GPU4 allocation内：

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  smoke "$PWD/work/h2d-two-readers/smoke-v1"
```

先返回 `smoke-v1/result/streams-compact.txt`。第一轮先检查实际stream及活动覆盖，不自动续跑。
失败回同一报告；未生成则给run.log最后30行。不需要SASS重提取，未新增CUDA kernel。

核对smoke后main命令：

```bash
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  main "$PWD/work/h2d-two-readers/main-v1"
```

回 `main-v1/result/streams-compact.txt`；完整streams.json和三份trace留本地。
只运行本轮，不追加CTA8×2、同地址、其他GPU或服务测试。

本地仅CPU解析测试与CLI失败报告检查；真实CUDA stream/graph行为待实机smoke验证。
