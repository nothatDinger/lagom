# 第十轮：真实 gather 的 NCU 计数器诊断

分支 `perf/hisparse-h2d-diagnosis`，继续基于 hash_verify；物理 GPU4/5、NUMA1、原生 venv。
当前只执行 §2 工具兼容性 smoke 与指标查询，返回紧凑结果后再决定正式计数器集合。

## 1. 依据、目标与工具条件

第九轮完整门禁通过，但 miss100 无 profile 的 ca_v2/legacy 仅改善约 4.2% / 6.1%，
ca_v2/ca_pair 为 +0.22% / −0.17%；向量化尚无可晋升服务的增量收益。
HBM 下 ca_v2/ca_pair 的 activity 却下降约 6.1% / 6.6%，不能说 GPU 执行无影响。
详见 [第九轮 §5](README_h2d_round9.md#5-正式块反馈与当前决定2026-10-14)。

接下来保持真实 legacy kernel、相同 host pool 和 plan，从两个维度推进证据：

- L1/L2 读请求/sector、可用的 system-memory aperture 读计数：同样逻辑 bytes 是否产生不同读取量？
- eligible/active warps、long scoreboard、load queue throttle：读取量接近时，慢卡是否有不同等待特征？

sector 不等于 PCIe TLP，L2 读计数也不必等于线上传输字节。long scoreboard 不只包含
host load，还可能包含 GPU plan 等 load；其百分比可能随 kernel 变长而改变，不能单独证明
某次 load 的绝对延迟或硬件队列深度。HBM 对照用于辅助解释，仍不做时间相减。

已确认 `/usr/bin/ncu` 为 **2022.4.1.0**，`RmProfilingAdminOnly: 0`。
该驱动参数允许非管理员使用计数器，但版本/驱动/当前 CUDA 13 JIT 产物的实际兼容性尚待验证。
不先升级工具或修改驱动；若 smoke 报兼容性错误，保留错误后再处理。

## 2. 当前执行：只捕获一次现有 eager kernel

复用 `bench_h2d_pair.py`，不新增 CUDA kernel 或捕获逻辑。过滤器只允许一个
`copy_cache_planned_kernel`：它应是第一项 host legacy / CUDA device0 / bank0 的 eager
正确性调用。该调用位于 JIT 编译之后、graph capture 之前；不是正式 profile 测量波。
只检验 NCU 能否捕获已知 CTA4/1024 kernel 和返回有效硬件计数，**不使用 smoke 带宽或时延**。
只捕获第一卡不代表正式双卡采集已经验证。

在正常属主账号中同步代码并设变量；NCU 命令放在 gpuq 作业内部，物理 GPU4/5 均已分配时运行。
若工作区仍保留已合并的两处 harness 改动，先保存改动再同步，避免覆盖本地工作。

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R10="$LAGOM_REPO/work/h2d-round10"
mkdir -p "$H2D_R10/ncu-smoke"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  /usr/bin/ncu \
  --kernel-name-base function --kernel-name 'regex:.*copy_cache_planned_kernel.*' \
  --launch-count 1 --replay-mode kernel --cache-control all --clock-control none \
  --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg,launch__grid_size,launch__block_size \
  --export "$H2D_R10/ncu-smoke/capture" \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --device-source-control \
  --batch 2 --misses 2 --plan-stride 128 --buffer-slots 128 \
  --source-mib 4 --max-banks 4 --bulk-ce-mib 1 \
  --warmup 1 --warm-ms 20 --epochs 1 --phases compute \
  --output-dir "$H2D_R10/ncu-smoke/pair" \
  >"$H2D_R10/ncu-smoke/ncu-run.log" 2>&1
```

使用全新目录；不要覆盖旧 report。命令不加 `--profile`，避免 PyTorch profiler 与 NCU
同时使用 CUPTI。NCU 使用 kernel replay、每 pass 清 cache、禁用 NCU 的时钟控制；
这些条件与生产不同，不能把结果当作性能主口径。`--launch-count 1` 只限制被 NCU
采集的 kernel 数；整个应用仍会完成原有 eager/graph 正确性校验。

若退出非 0，或日志出现 unsupported CUDA/PTX/driver、ERR_NVGPUCTRPERM、
无匹配 kernel 等错误，停在此处，只返回命令退出码与日志相关尾部；不运行全指标采集。
保留所有原文件，不据错误猜测 GPU/host 行为。若 CLI 不接受某选项，返回该项与 `ncu --help`
对应内容；本地没有 NCU/GPU，不能把 CPU 测试声称为这些 CLI/实机兼容性验证。

成功后导出原始指标，并查询可用的完整 metric 名称。查询保留在本地，不要求上传整份列表：

```bash
/usr/bin/ncu --import "$H2D_R10/ncu-smoke/capture.ncu-rep" \
  --page raw --csv --units base >"$H2D_R10/ncu-smoke/raw.csv"

CUDA_VISIBLE_DEVICES=4,5 /usr/bin/ncu --query-metrics --query-metrics-mode all \
  >"$H2D_R10/ncu-smoke/metrics-list.txt" \
  2>"$H2D_R10/ncu-smoke/metrics-query.err"

"$LAGOM_PY" scripts/pd_test/summarize_h2d_ncu_smoke.py \
  --pair-json "$H2D_R10/ncu-smoke/pair/pair.json" \
  --ncu-csv "$H2D_R10/ncu-smoke/raw.csv" \
  --metrics-list "$H2D_R10/ncu-smoke/metrics-list.txt" \
  >"$H2D_R10/ncu-smoke/ncu-compact.txt"
```

导出或查询失败分别反馈对应错误，不重跑 GPU 采集。摘要脚本只读文件，使用 Python
标准库，不导入 torch、不初始化 CUDA，也不修改原始数据。

本次回传约十行 `ncu-compact.txt`，另确认共享 host pool 在 node1。门禁：

- NCU 与应用退出 0，`compatibility_ok=True`，原始 `.ncu-rep` 存在。
- `PAIR` 的 records/payload/graph **8/8**、complete=True，Portable=1；无 PyTorch profile。
  配置是单个 miss 档 × sm/HBM/CE × 两卡，加两项 bulk CE，不是第九轮的 38 项。
- 目标为唯一 copy kernel，grid=4、block=1024，GPU duration 与 SM cycles 有效且正。
- `DEVICES` 仍为物理 GPU4/5；核对目标 launch 的 device/context 与第一项 eager 调用相符。
  NCU 原始 Device 字段只原样输出，不把它擅自解释成 NVML 物理编号。
- `METRICS_AVAILABLE/MISSING` 仅是实机工具暴露情况；缺少候选指标不等于其数值为零。

不要求再跑第九轮 correctness、paired smoke、反序或服务测试。本轮尚未安排完整 NCU
双卡采集；取得兼容性与支持列表后再选择小集合，明确 replay/cache 条件及匹配方式。
