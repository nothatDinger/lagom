# 第十轮：真实 gather 的 NCU 计数器诊断

分支 `perf/hisparse-h2d-diagnosis`，继续基于 hash_verify；物理 GPU4/5、NUMA1、原生 venv。
首次 NCU 2022.4 smoke 在准备 kernel 时失败。当前先按 §2 核对已有工具安装，
选定支持 CUDA 13 的版本后再复测同一 smoke；暂不改 replay/cache 参数。

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
该驱动参数放开管理员限制，但实际采集仍受工具、驱动及 kernel 产物兼容性影响。

### 首次 smoke 的事实与版本核对（2026-10-14）

gpuq `job-20261014-064638-cf009b9e` 失败，输出在该作业的 `stdout.log`，没有独立
`ncu-run.log`。NCU 连接 PID302007 并匹配到 `copy_cache_planned_kernel`，随后报告
`Failed to prepare kernel for profiling`，关闭应用，报应用错误码 9；没有 kernel 计数
或 `.ncu-rep`，输出目录只有空的 pair 子目录。这不是 GPU/host 性能证据。

官方版本记录已核对：

- [NCU 2022.4](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2022-4.html)
  新增 CUDA Toolkit 12.0 支持；
- [NCU 2025.3](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2025-3.html)
  新增 CUDA 13.0 支持；
- [CUDA 13.0 的组件表](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
  列出随附 NCU 2025.3.0.19。

版本支持范围不匹配是优先验证的假设，错误字符串本身还不能证明是 SASS 解码、
指令插桩还是其他准备步骤失败。H100 支持与 CUDA 13 产物支持也不是同一个问题。
下一次优先只换 NCU 完整安装的可执行路径；不更换 driver、Python、编译器或 JIT 源码，
不清空 JIT cache。若新版通过，只能先确认旧工具条件造成阻塞，不能宣布 H2D 根因已找到。

本轮不直接尝试三个建议选项，依据如下：

1. [Application Replay](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-replay)
   改变多 pass 的重跑/内存恢复方式，不保证绕过准备 kernel 的失败。当前 harness 还会在
   结束时写固定 `pair.json`，并在启动时拒绝已有该文件；若第二个 application pass 启动，
   会因此被拒绝。需要该模式时先适配输出与确定性检查，不能仅替换一个参数。
2. [CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#profile)
   中 `cache-control` 默认就是 `all`，删除选项不会构成新对照；改为 `none` 才改变行为。
3. 日志已经在连接的 PID302007 中定位到目标 kernel。末行 child-process 提示不证明
   漏采了子进程，本 harness 的 copy 在 Python 主进程执行；暂不加 `--target-processes all`。

另修正一处尚未执行到的导出命令：`--units` 已在 NCU 2022.4 移除，应使用
`--print-units base`。离线摘要增加 raw CSV 宽表（metric 列 + 单位行）支持，保留长表
支持及唯一 kernel/launch、完整指标、正值校验；不会把缺失计数当作零。

## 2. 先选工具，再复测同一 eager kernel

### 2.1 先只读检查已有安装

在实际 gpuq 环境可访问的位置检查候选，不先安装或修改全局 PATH：

```bash
for h2d_ncu_candidate in \
  /usr/local/cuda-13*/bin/ncu \
  /usr/local/cuda-13*/nsight-compute*/ncu \
  /opt/nvidia/nsight-compute/*/ncu \
  /opt/nvidia/nsight-compute*/ncu; do
  if [ -x "$h2d_ncu_candidate" ]; then
    printf 'NCU_CANDIDATE %s\n' "$h2d_ncu_candidate"
    "$h2d_ncu_candidate" --version
  fi
done
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
```

优先选择已有 **2025.3.x**（CUDA 13.0 对应版本），返回路径、版本及 driver 号即可。
若只有旧版本，反馈未找到；后续可以单独部署官方 NCU 到 work 下，不覆盖 `/usr/bin/ncu`
或改驱动，具体包与路径另行确定。工具发布页的 recommended driver 不等于最低 driver
要求，不能仅因不满足推荐号就要求升级驱动。

### 2.2 匹配版本选定后才执行以下一次复测

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
# 先将 H2D_NCU 设为 §2.1 确认的新版 ncu 绝对路径；不使用旧 /usr/bin/ncu。
: "${H2D_NCU:?Set H2D_NCU to the verified CUDA-13-compatible ncu path}"
export H2D_NCU_RUN="$H2D_R10/ncu-smoke-supported"
if [ -e "$H2D_NCU_RUN" ]; then
  printf 'Use a new output directory: %s\n' "$H2D_NCU_RUN" >&2
  exit 1
fi
mkdir -p "$H2D_NCU_RUN"
"$H2D_NCU" --version >"$H2D_NCU_RUN/ncu-version.txt"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$H2D_NCU" \
  --kernel-name-base function --kernel-name 'regex:.*copy_cache_planned_kernel.*' \
  --launch-count 1 --replay-mode kernel --cache-control all --clock-control none \
  --metrics gpu__time_duration.sum,sm__cycles_elapsed.avg,launch__grid_size,launch__block_size \
  --export "$H2D_NCU_RUN/capture" \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --device-source-control \
  --batch 2 --misses 2 --plan-stride 128 --buffer-slots 128 \
  --source-mib 4 --max-banks 4 --bulk-ce-mib 1 \
  --warmup 1 --warm-ms 20 --epochs 1 --phases compute \
  --output-dir "$H2D_NCU_RUN/pair" \
  >"$H2D_NCU_RUN/ncu-run.log" 2>&1
```

使用全新目录；不要覆盖旧 report。命令不加 `--profile`，避免 PyTorch profiler 与 NCU
同时使用 CUPTI。NCU 使用 kernel replay、每 pass 清 cache、禁用 NCU 的时钟控制；
这些条件与生产不同，不能把结果当作性能主口径。`--launch-count 1` 只限制被 NCU
采集的 kernel 数；整个应用仍会完成原有 eager/graph 正确性校验。

若退出非 0，或日志出现 unsupported CUDA/PTX/driver、ERR_NVGPUCTRPERM、
无匹配 kernel 等错误，停在此处，只返回命令退出码与日志相关尾部；不运行全指标采集。
保留所有原文件，不据错误猜测 GPU/host 行为。若 CLI 不接受某选项，返回该项与 `ncu --help`
对应内容；本地没有 NCU/GPU，不能把 CPU 测试声称为这些 CLI/实机兼容性验证。

成功后仍使用同一个新版 NCU 导出与查询，避免采集用新版、导出又落回 PATH 中旧版。
查询保留在本地，不要求上传整份列表：

```bash
"$H2D_NCU" --import "$H2D_NCU_RUN/capture.ncu-rep" \
  --page raw --csv --print-units base >"$H2D_NCU_RUN/raw.csv"

CUDA_VISIBLE_DEVICES=4,5 "$H2D_NCU" --query-metrics --query-metrics-mode all \
  >"$H2D_NCU_RUN/metrics-list.txt" \
  2>"$H2D_NCU_RUN/metrics-query.err"

"$LAGOM_PY" scripts/pd_test/summarize_h2d_ncu_smoke.py \
  --pair-json "$H2D_NCU_RUN/pair/pair.json" \
  --ncu-csv "$H2D_NCU_RUN/raw.csv" \
  --metrics-list "$H2D_NCU_RUN/metrics-list.txt" \
  >"$H2D_NCU_RUN/ncu-compact.txt"
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
