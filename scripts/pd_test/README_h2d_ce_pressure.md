# GPU4：分档 CE 平均压力与 SM 吞吐

目标：在CE流量逐步增加时，观察SM吞吐曲线，区分解释的优先级。
**CPU限速形成16MiB传输脉冲，不是硬件匀速；低平均流量下SM变慢也可能来自
脉冲期间的瞬时链路竞争，不能凭此直接断言请求队列瓶颈。**

## 固定条件与执行

沿用SM/CE独立地址布局、同一legacy JIT、GPU4/NUMA1。SM为CTA4/1024，
每bank B32/miss100，每SM计时块64 banks。CE每次连续16MiB。

CE目标平均压力5/15/30 decimal GB/s，按CPU deadline周期发起单次CE graph。
最多一笔CE在途，错过期限不补发、不堆队列。无GPU sleep、自旋或忙等限速kernel。
实际压力必须从GPU事件统计，不把目标值当实际值。

SM使用单次提交的长graph，含smoke32/main128个计时块及边界事件，避免CPU限速
导致SM提交间隙。每段截去前后各20%，中心窗口按完整SM计时块边界收紧；
SM字节/中心窗口时间给出有效吞吐；CE跨边界脉冲的字节无法精确切分，因此报告
全部包含脉冲的字节下界和所有相交脉冲的字节上界。两路同一时间窗口。
保留每个SM块及CE脉冲的GPU事件区间供后续只读分析。

每档执行OFF-before / ON / OFF-after。main四轮，轮间反转档位顺序；
smoke一轮三档。每段预热250ms，NVML只采段前/后，计时内不采NVML/Rx。
SM保持率=ON吞吐/前后OFF吞吐均值。所有段后核对SM与CE输出字节。
额外每档一份profile，仅验证真实SM/CE活动、设备、grid、bytes及stream；
profile可能扰动CPU节拍，不将其速率冒充无profile压力。

## 门禁

- 身份GPU4、host NUMA1、source_disjoint/independent_outputs为True；legacy JIT一致。
- 前置字节检查checks=8/8；每个压力窗口correct=True。
- CE pilot≥45GB/s；结果中CE_calibration=True。
- 无profile每窗valid=True：SM提交占比≤10%，SM计时块事件覆盖≥98%。
- ON中心至少4个完整CE脉冲；实际CE速率上下界均在目标±20%内。
- 每档前后OFF的SM吞吐比0.9–1.1；各窗状态无错误、P0、Gen5x16。
- profiles=3/3，SM/CE activity数量、设备、grid/bytes/stream全部符合。
- success=True、results_valid=True。不以吞吐升降或重叠率高低作为门禁。

在gpuq物理GPU4 allocation内：

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  smoke "$PWD/work/h2d-ce-pressure/smoke-v1" sm-ce-rate
```

smoke全部门禁满足后直接续跑：

```bash
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_streams.sh \
  main "$PWD/work/h2d-ce-pressure/main-v1" sm-ce-rate
```

反馈smoke门禁一行和main/result/streams-compact.txt全文。
streams.json与三份trace留本地。任何门禁失败停止，回compact及run.log最后30行，
不自行放宽速率门槛或增加实验。本地无GPU，真实capture/事件行为须由smoke验证。

## 判读边界

先看实际CE平均速率、事件占空比及SM保持率，确认档位确实拉开、OFF未漂移。
若低压力SM基本保持、高压力才明显下降，支持共同容量因素在高压力下更重要；
若低平均压力也下降，需结合脉冲占空比和时间线，不能据此排除瞬时带宽限制。
三点曲线用于筛选解释，不足以识别PCIe tag/TLB/硬件锁，也不证明链路线速饱和。
已有满压SM+CE作为背景，本轮不把不同轮次精确拼成统一容量曲线。

## smoke-v1 capture 修复

旧版本在长graph capture内调用已有SM graph.replay()，实机返回
cudaErrorStreamCaptureUnsupported，随后capture被invalidated。这发生在窗口构造，
没有有效压力测量；与kernel数据正确性或带宽无关。

修复为直接捕获同一Reader.launch_bank的legacy kernel发射，
smoke 32×64=2048、main 128×64=8192条kernel节点，每64条两端记录时间。
只在capture结束后replay整个长graph一次。内核、bank顺序、源/目的、CTA和字节计数不变。
报告capture_kind=flat_kernel_launches及sm_kernel_nodes；profile仍核对实际kernel数量。

重新使用新目录smoke-v2；全部门禁通过后使用main-v2，不覆盖失败的smoke-v1。
