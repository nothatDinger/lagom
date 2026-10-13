# 第一轮：真实 decode/verify 的 KV H2D 测量

基线：`hash_verify@0fef460dd95a70eb3d921cc239bc989d1f2842ae`。
所有改动在新分支 `perf/hisparse-h2d-diagnosis`。
本轮建立测量事实，暂不改生产 copy 并行度、缓存算法或调度。
历史 23/55 GB/s 不作为定量依据。

首轮实机修复：probe 现在按 request pool 的完整 slot 数预分配，覆盖启动/warmup 的 batch=256，
不再按 `cuda-graph-max-bs-decode=64` 推断上限。`PROBE_MAX_BATCH` 是额外预留的下限，较小的旧值会自动提升。
collector 兼容 `/start_profile` 的纯文本 `Start profiling.`，并保存 HTTP 响应。
多 stream 的旧 trace 现在可以离线分析，不因 stream ID 数量直接拒绝；已有 C/D 文件无需重新采集。

## 1. 本轮回答的问题

1. 真正的每层、每个 verify window、每 rank H2D 有多少有效字节？
2. 相应 copy kernel 在 GPU 上执行多久？带宽是否随负载、层或 rank 变化？
3. 正常 residency 与强制 H2D 的负载有何差别？copy 在整个 decode 时间线中占多少，规划与等待在哪里？

优先测**正常缓存**：`force_h2d_each_step=false`。
另跑 `true` 作为受控对照，复现附件里的强制搬运设置。两组分别保存和分析。
强制模式仍会在同一个 verify window 内去重；不能按 `B*W*K*584` 直接算实际字节。
始终保留默认 CUDA Graph。暂不加入 skip-IO，也不比较失效输出的 DSPARK acceptance。

## 2. 测量定义与校验

这里的 copy 是 SM 直接读取 pinned host KV，代码位于
`python/sglang/kernels/jit/csrc/hisparse.cuh` 的 `copy_miss_item` 和
`copy_cache_planned_kernel`，不是 CE/DMA `cudaMemcpyAsync`。
DSv4 C4 的有效载荷是 **576 value + 8 scale = 584 bytes/entry**，不含 page padding。

- 实际字节：只读 `num_real_reqs` 范围内的真实 `miss_count`，每次 copy 的 `bytes=sum(misses)*584`。
- copy 带宽：`sum(bytes)/sum(copy_gpu_us)/1000`，单位十进制 GB/s；不会平均逐次 GB/s。
- copy 时间：从每个真实 GPU copy kernel 的 trace 时长得到，不包含 planner、主机提交间隙或 recorder。
- 多 stream 时另报 `copy_gpu_union_us` 和 `copy_active_wall_GBps`：重叠 copy 的时间区间取并集。
  `copy_GBps` 仍使用 kernel 时长之和，两个口径不能混用；出现多个 stream ID 本身并不证明发生并发。
- 整个 decode 窗口的有效带宽：只有选定**完整稳态 decode 迭代边界**后才用相同窗口的 bytes/wall time 算。
- TP rank 分别报告。总线物理流量、CE 带宽、有效 KV 载荷带宽分别解释。
  KV 载荷计数不能替代 PCIe wire counter，也不能排除 GPU cache 的影响。
- 每次 copy 后有一个 GPU recorder；控制标志和计数器地址固定，CUDA Graph 每次 replay 都记新记录。
  循环内没有 `.item()`、`.cpu()`、CUDA event 同步或逐次日志。
  只在 profile 开始/结束同步并导出；这些边界不用于性能基准。
- recorder 是额外 kernel，可能改变时间线。报告单独给出其耗时，并要求下面的无 profile A/B 对照。
  profiler 对主机提交的影响也必须考虑，不能把 profile run 的 TPOT 当作性能结果。
- 严格按 stream 配对 recorder 与紧邻它的 copy 节点。recorder 区间不重叠时，其执行顺序保证全局计数顺序，
  可得到逐次和逐层时间；区间重叠时只给整段载荷/时间总量，不猜测每条记录与哪次 copy 对应。
  后者不支持切割时间窗口或只选部分 B/W 的带宽，也不输出逐层时间。
  溢出、缺少 graph-node 时间、配对错误、skip-IO 或字节不一致仍拒绝输出 GB/s。

当前覆盖 **DSv4 speculative 同步 verify** 路径，包括静态 graph 和 ragged eager window。
其他模型/普通 decode 不自动套用此测量。

## 3. 导入分支与实机先验检查

优先从远端取新分支，在已有仓库执行，保留原 checkout：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git worktree add -b perf/hisparse-h2d-diagnosis ../lagom-h2d-round1 FETCH_HEAD
cd ../lagom-h2d-round1
```

离线传输时也可以用 bundle 替代远端 fetch：

```bash
git fetch origin hash_verify
git fetch /path/to/hisparse-h2d.bundle \
  refs/heads/perf/hisparse-h2d-diagnosis:refs/heads/perf/hisparse-h2d-diagnosis
git worktree add ../lagom-h2d-round1 perf/hisparse-h2d-diagnosis
cd ../lagom-h2d-round1
```

完成任一种导入后，继续设置环境和检查：

```bash
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$(dirname "$LAGOM_PY"):/usr/local/cuda-13.0/bin:/usr/bin:/usr/local/bin:$PATH"
export H2D_OUT="$LAGOM_REPO/work/h2d-round1"
export SHAREGPT_PATH=/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json
mkdir -p "$H2D_OUT"
"$LAGOM_PY" -c 'import sglang, torch; print(sglang.__file__); print(torch.__version__, torch.version.cuda)'
CUDA_VISIBLE_DEVICES=4 "$LAGOM_PY" -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hisparse_h2d_probe.py \
  >"$H2D_OUT/smoke.txt" 2>&1
cat "$H2D_OUT/smoke.txt"
```

必须看到 **3 passed**，不能只有 skips；否则先返回 smoke.txt，暂停服务测量。
检查内容包括每次 graph replay 计数、padding 排除、溢出检测，以及真实 paged C4 copy 的字节正确性。
当前开发环境只有 CPU，CPU 分析测试与 H100 SM90 离线编译已验证，实机执行仍需这一步。

## 4. 服务启动：复用附件的 PD 拓扑

停掉你原有的同端口实验服务，然后从本 worktree 启动。此处 `FORCE_H2D=false` 是主实验；
以后改成 `true` 单独做受控对照。CUDA_VISIBLE_DEVICES 本身不保证 CPU/host pool 的 NUMA 亲和，
本轮先记录环境，不增加新的绑核/绑内存配置。

```bash
export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1
export SGLANG_PROFILE_V2=0
export SGLANG_DEBUG_HISPARSE_SKIP_IO=0
unset SGLANG_DEBUG_HISPARSE_COUNT_BYTES SGLANG_DEBUG_HISPARSE_PROBE_VOLUME
MODEL_PATH=/mnt/public_data/deepseek-ai/DeepSeek-V4-Flash-0731/
FORCE_H2D=false
COMMON=( --model-path "$MODEL_PATH" --tp 4
  --moe-runner-backend flashinfer_mxfp4 --cuda-graph-max-bs-decode 64
  --hisparse-config "{\"force_h2d_each_step\":$FORCE_H2D,\"top_k\":512,\"device_buffer_size\":3072,\"host_to_device_ratio\":5}"
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 )

CUDA_VISIBLE_DEVICES=0,1,2,3 SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0 \
  "$LAGOM_PY" -m sglang.launch_server "${COMMON[@]}" \
    --mem-fraction-static 0.80 --radix-eviction-policy lru \
    --port 30001 --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device '{"0":"mlx5_0","1":"mlx5_3","2":"mlx5_4","3":"mlx5_5"}' \
    >"$H2D_OUT/prefill.log" 2>"$H2D_OUT/prefill.err" &

# A: capacity=0. B/C: restart decode with capacity=16384. Keep everything else identical.
CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0 \
  SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH=1024 \
  "$LAGOM_PY" -m sglang.launch_server "${COMMON[@]}" \
    --mem-fraction-static 0.85 --enable-hisparse --disable-radix-cache \
    --port 30000 --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device '{"0":"mlx5_6","1":"mlx5_7","2":"mlx5_8","3":"mlx5_9"}' \
    >"$H2D_OUT/decode.log" 2>"$H2D_OUT/decode.err" &

"$LAGOM_PY" -m sglang_router.launch_router --host 127.0.0.1 --port 30002 \
  --pd-disaggregation --mini-lb \
  --prefill http://127.0.0.1:30001 --decode http://127.0.0.1:30000 \
  >"$H2D_OUT/router.log" 2>"$H2D_OUT/router.err" &
```

等三个 `/health` 都正常。保存完整实际启动命令；如果安装的 router 参数名不同，使用你原来已经跑通的 router 命令。
不要把旧日志覆盖掉：每次重启使用不同结果目录。
新版也可 unset `SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH`，自动使用实际 request pool slot 数。
已有实验使用 1024/16384 的环境设置可以保留，避免改变观察者开销对照。

负载使用附件同样的 ShareGPT 派生 random 数据、seed=0、64 请求、并发上限 32。
脚本增加 `--tokenize-prompt`，直接传 input_ids，避免文本 decode/encode 改变 16384-token 输入长度；
`random-range-ratio=1` 固定长度，默认 ignore-EOS 固定输出 4096 tokens。
检查 benchmark 的 completed=64、total_input_tokens=1048576、total_output_tokens=262144；
不满足时先返回错误和日志，不将不同负载混算。客户端并发 32 与 GPU 实际 batch 仍需分别校验。

## 5. 执行顺序

**A：正常 residency、probe=0、无 profile。** 使用相同脚本重复三次：

```bash
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_OUT" normal-A1
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_OUT" normal-A2
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_OUT" normal-A3
```

**B：重启 decode，改为 probe=16384，其余相同，无 profile。** 同样重复三次，用 `normal-B1/B2/B3` 标签。
不开 profile 时 recorder 仍是一个空 kernel，因此 B 衡量增加 graph 节点的成本；
采集激活后的计数写入成本由 C 的 recorder 时长补充评估。
若 A/B 的 TPOT 差异明显超过重复运行波动，先分析观察者开销，不据此宣称根因或收益。

**C：保持 B 服务，另跑一次负载，在稳态直接对 decode 端点采集。**
终端 1：

```bash
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_OUT" normal-profile
```

终端 2：等 decode 日志显示实际 `bs=32` 已稳定，且仍有足够生成 token 后执行：

```bash
"$LAGOM_PY" scripts/pd_test/collect_h2d_environment.py --output-file "$H2D_OUT/environment.json"
"$LAGOM_PY" scripts/pd_test/collect_h2d_profile.py \
  --output-dir "$H2D_OUT/normal-trace" --steps 64
```

这里 64 是 scheduler forward 次数，不能视为 64 个输出 token；实际 B/W/每层 copy 次数由数据校验。
若没有稳定 B=32，原样返回日志和实际 B 分布，不按客户端 max-concurrency 假定 GPU batch。
等四个 decode TP rank 都显示 `Profiling done`。每 rank 会有 `.trace.json.gz` 和 `.h2d.json`。

**D：强制 H2D 对照。** 单独结果目录；重启相关服务，将 COMMON 的 force 改为 `true`、decode probe=16384。
重复 C，使用 `forced-profile` 标签和 `forced-trace` 目录。保留其 benchmark JSONL 和日志，但不拿该 profile run 作吞吐基准。
这组用于区分小负载启动成本与持续传输成本；以后有需要再做其完整无 profile 吞吐基准。

## 6. 本地分析与 trace 兼容性

为每个 rank 选取**同一 capture 的**两个文件：

```bash
"$LAGOM_PY" python/sglang/benchmark/hisparse_h2d.py \
  --probe /path/to/TP-0-PID-xxx-GPU-0.h2d.json \
  --trace /path/to/TP-0.trace.json.gz \
  --output-file "$H2D_OUT/normal-TP0.json"
```

先不加 B/W 过滤，保留实际分布。需要时可加 `--batch-size 32 --verify-width 6`。
多 stream 会在 `pairing` 中列出实际 stream ID 和 recorder 是否重叠；若重叠，
`mode=capture_totals_only` 下仍可看整段带宽，但 layer/geometry 只有载荷统计。
只看 probe 的载荷分布，不计算 GB/s：

```bash
"$LAGOM_PY" python/sglang/benchmark/hisparse_h2d.py \
  --probe /path/to/D-TP0.h2d.json --probe-only \
  --output-file "$H2D_OUT/D-TP0-payload.json"
```

不要把四个 rank 的时长累加计算单卡 GB/s。
`--window-start-us/--window-end-us` 只接受自行确认的完整稳态迭代边界；无需第一轮手工选择，返回原始 trace 后再一起核对。
torch trace 和 probe 内还带同一 capture ID/PID，解析器检查两者匹配，避免不同轮次恰好节点数相同被混用。
解析器严格拒绝只有 `cudaGraphLaunch` 而没有内部 kernel 的 trace。

如果 torch trace 缺少 graph nodes，保留错误信息，采用 Nsight 重采，不切换 eager 冒充同一个结果。
用原 decode 命令前加（probe 仍为 16384）：

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output "$H2D_OUT/normal-nsys" "$LAGOM_PY" -m sglang.launch_server ...
```

达到相同稳态后，collector 加 `--activity nsys`，并使用新的 capture 目录。
完成后 `nsys export --type sqlite --output normal-nsys.sqlite normal-nsys.nsys-rep`。
分析器的 `--trace` 也支持该 `.sqlite`。SQLite 不含 torch 的 capture ID 元数据，务必使用同一 capture 的 probe；
同样严格校验节点与计数，不静默修正丢失的数据。
Nsight 版本不支持这些参数时，先返回 `nsys --version` 与错误。

## 7. 请返回的数据

- `smoke.txt`、`environment.json`、实际 SHA 和完整启动命令。
- A1–A3、B1–B3 的 JSONL/压测日志；C、D 的 JSONL 用于核对工作负载和 acceptance。
- C、D 各四个 decode rank 的 `.h2d.json`、对应 `.trace.json.gz`，或 Nsight SQLite/原始报告。
- 各次 decode.log/decode.err，至少覆盖稳态、profile 起止与最终统计。
- 四个 rank 的分析结果，或分析器的完整错误。如果文件大，先发正常 C 的 TP0 两文件，其他 rank 保留。

收到后先核验：真实 B/W、每层样本数、miss 计数与 padding、计时边界、四 rank 一致性和观察者开销。
再定位 copy、hash gather/commit、force invalidation、attention、CPU 提交与 PD 通信在时间线中的关系。

## 8. 后续实验由本轮数据决定

已确认代码：`HiSparseCoordinator._init_shared_index_prefetch` 把 copy 配置固定为 4 blocks，
`_run_swap_in_verify_window` 在当前 stream 上用它完成 DSPARK 的同步 copy；wrapper 默认 1024 threads/block。
即 128 warps，但单次 launch 的 CTA 只能占用至多 4 个 SM。
这个设计原本用于在 side stream 与计算重叠时限制 SM footprint，是否适合当前同步大负载是待验证假设。

- copy 本身慢且 payload 足够大：用匹配 miss 量的 copy-only replay 扫描 CTA 数，再做真实服务 A/B，检验因果。
- copy 快而 decode 窗口速率低：沿 planner、串行依赖、提交间隙或 PD 等待继续定位。
- 小 payload 时带宽低但 copy 绝对时间很小：先评估优化能降低多少 decode 时间，避免按 GB/s 排名。
- rank/节点差异明显：再采集实际 host pool NUMA placement，做严格匹配的绑内存与并发对照。

CE/SM 平台上限将用相同 GPU、并发、NUMA 和工作集另测。连续 DMA 的结果不是离散 584-byte SM gather 的硬上限；
也不会把当前观测到的连续 SM 数字当作架构不可超越的上限。
