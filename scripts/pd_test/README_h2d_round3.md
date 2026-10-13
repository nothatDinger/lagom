# 第三轮：同一 host 分配、双卡交替、状态与计时校验

已完成 pair-main（未开启 profile）后的下一步见 [第四轮：补齐实际活动时长](README_h2d_round4.md)。
第三轮的 correctness 与无 profiler 性能结果有效，无需重新跑 smoke。

分支仍为 `perf/hisparse-h2d-diagnosis`，生产 copy 参数不变，原生 venv 执行。
GPU4/5 必须空闲；本轮不启动模型，不使用 GPU0–3，不改时钟、功耗上限或驱动设置。

## 1. 第二轮支持的结论和仍缺的证据

用户反馈：smoke 成功，独立测试的 host buffer 全部在 NUMA1，单卡结果如下（十进制 GB/s）：

| 载荷/方法 | GPU4 | GPU5 |
|---|---:|---:|
| 1,868,800 B，SM CTA4 | 28.94 | 17.46 |
| 同载荷，连续 CE | 47.42 | 24.09 |
| 23,771,136 B，SM CTA4 | 32.42 | 23.92 |
| 同载荷，连续 CE | 54.76 | 54.60 |
| 64 MiB，连续 CE | 55.24 | 55.08 |

- 相同 NUMA node、脱离服务仍出现 SM 差异，说明产生差异不需要服务内的 rank 并发或跨 node 分配。
  原服务已停止，实际 KV host pool 的 live 证据尚未取得，因此不能声称它的放置已经验证。
- CTA4 在已测试的 CTA4/8/16/32/64、block_size=1024 范围内最好。不能推成所有 block-size/grid 的全局最优，
  也暂时不应在生产里增大 CTA。
- 大 CE 的约 55 GB/s 确认 GPU5 没有明显更低的 bulk 传输上限；小 CE 却差约两倍。
  它不能排除固定成本、运行状态、计时边界和路径延迟，也不能直接证明某个 switch hop 是根因。
- 从报告带宽换算，小 SM 的 event 区间约 64.6/107.0 µs，小 CE 约 39.4/77.6 µs：
  GPU5 在两条小载荷路径都多约 40 µs。这个共同异常应先拆分，再解释 SM 独有的部分。
- 旧测试两卡分别创建进程和 host allocation；绑定到同一个 node 不等于使用相同物理页。
  旧 benchmark 的 GPU events 虽然在 graph 内，也仍会包含两个 event 节点之间的 GPU 调度/等待，
  不能直接当作 profiler 的实际 copy 活动时长。

## 2. 新对照的范围

`bench_h2d_pair.py` 在一个进程中分配一份 mmap host buffer，固定源数据和 miss plan。
两卡顺序读取同一份 pinned allocation；没有两卡同时搬运、没有 NCCL/Gloo。
顺序为 A→B→B→A，默认两个 epoch。可以以后用 `--reverse-order` 改为 B→A→A→B 验证顺序影响，先不要求追加。

为共享注册使用 `cudaHostRegisterPortable=1`，旧 solo 是 flags=0，差异明确记录。
注册 context 默认 CUDA logical 0，并可用 `--register-device` 设置；这只是实验控制，不修改生产 allocator。
内存固定在这次注册期间，两卡使用相同 host 对象和 CPU plan，避免不同分配作为混杂因素。
它还不能替代实际服务 miss plan 重放。

只测 CTA4/1024 threads，载荷仍为 miss100/miss1272 加 64 MiB CE。
每张卡、每个方法都校验首尾两个 bank 的全部 value/scale 或连续 copy 字节。
多卡 capture 显式指定各卡的 stream；capture 后清零输出再 replay 并校验最后一个 bank，
避免只验证 eager 正确、却未验证 graph 实际搬运。每波等待整个 replay stream 完成后再读取 event 和切换卡。

两个阶段：

1. `copy`：copy graph 预热后测量。
2. `compute`：先在当前卡执行约 250 ms 矩阵计算，再 copy graph 预热后测量。

矩阵计算位于 copy 计时之外。它可以改变 GPU 状态，也会改变 cache，所以阶段差异本身不能单独归因于时钟。
每波前后用 NVML 读取 SM/memory clock、P-state、PCIe gen/width、限速原因 bitmask、功率和温度。
按 UUID 匹配 NVML 物理卡，避免 CUDA logical index 混淆。
同时只读 sysfs 的 endpoint→root PCI 路径及 link speed/width，用于核查拓扑；缺字段保留错误，不能假定路径相同。
读取失败会记录 error/unknown，不把未知状态当成正常；前后快照不证明每条 copy 期间状态恒定。

无 profile 的结果用于性能比较。随后 `--profile` 单独测一次每个 graph，记录同一波的 event interval，
并从 torch trace 的实际 GPU kernel/HtoD activity 提取 copy 时间。
同步 CPU annotation 包围整波，严格要求 copy node 数与 banks 相同；CE 若提供 bytes 字段也会校验。
scope 必须是唯一的 `cat=user_annotation`。Kineto 可能输出同名 `gpu_user_annotation`，
它是 GPU 上的投影，不计作第二个 CPU scope，也不用它的时间范围替代 CPU 边界。
若仍有两个 CPU annotation，解析继续失败并打印候选元数据，不能任选一个。
缺节点、范围或 clock 数据自洽检查失败时，不输出猜测的 activity GB/s。
profiler 运行用于验证计时口径，不能代替无 profiler 的性能基线。

## 3. 执行：一次新 smoke，一次正式对照

在此前实验 worktree 中更新；确认没有本地 tracked 修改后 fast-forward：
仅离线恢复旧 smoke 时，执行前两条 git 命令后直接进入“先离线恢复”，可跳过 venv/NVML 预检。

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R3="$LAGOM_REPO/work/h2d-round3"
mkdir -p "$H2D_R3"
"$LAGOM_PY" -c 'import pynvml; print("NVML Python binding available")'
```

若最后一行提示缺少 pynvml，仅需为该 venv 安装 Python binding：
`"$LAGOM_PY" -m pip install nvidia-ml-py`。不安装或修改 GPU driver。

### 已有 smoke 因 `Expected one scope, found 2` 失败：先离线恢复

旧解析器只按名称和 `ph=X` 匹配，没有限定 CPU annotation；同名 GPU 投影可能被误算。
用户报告的重复事件类别尚需从 trace 确认，不能仅凭数量 2 推断成重复 replay 或两卡各发一次。

更新分支后，在仓库根目录执行以下命令，只读已有两个文件；系统 Python 即可，
不导入 torch，不创建 CUDA context，不需要空闲 GPU，也无需重新跑 smoke：

```bash
python3 scripts/pd_test/reanalyze_h2d_pair.py \
  --report work/h2d-round3/pair-smoke/pair.json \
  --trace work/h2d-round3/pair-smoke/pair.trace.json \
  --output-dir work/h2d-round3/pair-smoke-reanalyzed
```

输出到新目录，保留原文件；再次解析应使用另一个新目录。
`pair-compact.txt` 现在直接包含独立的 correctness 汇总、10 个 profile scope 的覆盖校验、
以及每个同名事件的 `cat/pid/tid/ts/dur` 等少量字段。
若类别为一个 `user_annotation` 加一个 `gpu_user_annotation`，就确认了此次误判来源。
离线输出 `pair-reanalyzed.json` 还记录原始输入 SHA256 和旧 profile 状态，便于复核。

必须退出 0、`ready_for_main=True`、`profile_success=True`，
且 `CORRECTNESS ... graph_replay_checked=10/10 complete=True`、profile coverage 为 10/10，
才能直接进入下面的正式对照，无需再跑 smoke。
否则只反馈新目录的 `pair-compact.txt`；如果未生成该文件，反馈错误末尾约 30 行。

`graph_replay_checked` 是 capture 后清零、replay、逐字节验证的结果，独立于 profiler 解析。
`profile_success=False` 不意味着 graph 正确性失败；旧摘要没有打印该字段，新摘要会直接展示。
当前 profile 固定为 5 个 case/method × 2 张卡 = 10 个 scope，仅测 `profile_compute`；
`copy/compute` 两阶段属于此前无 profiler 的对照，并不是这 10 个 scope 的另一倍数。

### 首次运行才需要 smoke

先 smoke：

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --batch 2 --misses 2 70 --plan-stride 128 --buffer-slots 128 \
  --source-mib 4 --max-banks 4 --bulk-ce-mib 1 \
  --epochs 1 --warmup 1 --warm-ms 20 --profile \
  --output-dir "$H2D_R3/pair-smoke" >"$H2D_R3/pair-smoke.log" 2>&1
```

必须退出成功，`pair-smoke/pair-compact.txt` 的 `success=True profile_success=True`。
完整 `pair.json` 的 correctness 全通过，两卡分别识别为物理 4/5；profile 的 10 个 scope 均有有效 activity。
紧凑摘要也会显示 `CORRECTNESS ... graph_replay_checked=10/10 complete=True`。
smoke 仅验证 Portable 共享、双卡 event/capture、NVML 映射和 profile 活动解析，带宽不用于性能判断。
本开发环境无 GPU；这些新增 GPU 路径需要该实机检查。
失败就反馈最后约 30 行错误和紧凑文件（若存在），不要继续。原始 trace 保留本机。

### 正式对照

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py --profile \
  --output-dir "$H2D_R3/pair-main" >"$H2D_R3/pair-main.log" 2>&1
```

约 512 MiB host memory，两张卡各几百 MiB 测试 buffer；不使用模型权重。
此轮仍不要求四卡并发、远端 NUMA、ShareGPT 或重启服务。
下次自然重启 decode 时，再补真实 KV host metadata；不要为已退出的 PID 继续采样。

## 4. 只需反馈 pair-compact.txt

`pair-main/pair-compact.txt` 已包含：

- 物理 UUID/PCI bus、共享 host 分配和 NUMA VMA、flags 与注册 context。
- 两阶段每卡各用例的加权 GB/s、mean/median/p95、ABBA 每波均值和状态范围。
- profiler 的 event/activity GB/s 与时间比。`UNASSIGNED` 会给缺节点/设备/时钟等原因。

只上传这个小文件；pair.json 和 pair.trace.json 都留在实机。

| 新结果 | 支持的下一步 |
|---|---|
| GPU5 小 CE 的 event 慢，但实际 activity 接近 GPU4 | 排查 graph/event/引擎调度间隙，不把它直接叫 PCIe transfer 慢 |
| compute 阶段差距收敛，且状态发生变化 | 继续区分状态/cache/空闲切换；不能仅凭预热效果宣布时钟根因 |
| 同 host、状态接近、activity 仍有稳定 SM 差距，而大 CE 接近 | 再查 SM sysmem 访问的事务放大、warp 等待、ATS/IOMMU 与 PCIe 路径/读请求行为 |
| 同 host 后差距消失或随顺序变化 | 优先检查原独立进程/分配、注册 context、缓存及系统负载，不归因物理卡固有差异 |

两卡能得到同一 bulk CE 上限，并不证明小传输延迟相同；增大 CTA 没收益，也不证明并行度对所有形状都无关。
本轮将先确定“哪个时间、在什么状态下、是否仍随物理卡变化”，再进入硬件访问机制的归因。
