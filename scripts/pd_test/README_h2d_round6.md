# 第六轮：检验源读取 cache policy，并准备回到原始 decode

目标仍是 `hash_verify`、HiSparse + DSPARK、ShareGPT 派生 16K 输入 / 4K 输出、B=32
在 8×H100 上的实际 decode H2D 和端到端吞吐。本轮提交第一个默认关闭的生产优化候选，
用既有 paired 工具检验；微基准通过后，同一个开关直接进入原始工作负载。
分支仍为 `perf/hisparse-h2d-diagnosis`，原生 venv，不要求 Docker。

## 1. 最新 A/B 的证据边界

- A：注册调用的 current context 从 GPU4 改到 GPU5，host SM 差距仍存在，CE 仍接近。
  这不支持“注册调用时选错 context”这一解释；没有反转所有初始化顺序。
- B：host SM activity 为 61.35 / 82.03 µs、728.21 / 990.51 µs；
  相同 plan、counts、目的 buffer 和 kernel 的本卡显存源对照为 31.41 / 31.58、284.85 / 281.39 µs。
  差距依赖 host-backed 源路径，不能解释成与源无关的固定 GPU 执行惩罚。
- 仍未证明是哪一种 host read 机制。共同的 plan/store 成本仍存在，其与源路径的交互也未排除。
  显存源与 host 源的瓶颈、cache 和重叠不同，不能相减当作 PCIe 耗时。

两处数值修正不改变判断：A 小载荷 `82.38/61.14=1.347`，不是 1.328；
B 大载荷显存源 `281.39/284.85=0.988`，约差 1.2%，可称接近而非严格在 1% 内。

## 2. 代码干预：只改变 planned-copy 的源读取策略

实际调用链是 `HiSparseCoordinator._run_swap_in_verify_window` →
`copy_cache_planned_mla` → `copy_cache_planned_kernel` → DSv4 `transfer_item`。
现有 CUDA 路径每 warp 搬一条 KV；DSv4 源读取是普通 C++ 的 64-bit load，随后写入 paged C4 buffer。
本次不改变 planner、miss 去重、warp 到 item 的分配、读取宽度、value/scale 布局、stores 或 CTA4/1024。

新增三种编译特化：

| 策略 | 源读取 | 用途 |
|---|---|---|
| `legacy` | 原来的 C++ load | 默认基线 |
| `ca` | 显式 `ld.global.ca.b64` | 控制改写成 inline PTX 的影响 |
| `cg` | 显式 `ld.global.cg.b64` | 检验绕过 L1 的读取策略能否降低耗时 |

`ca` 与 `cg` 的源码只差 cache modifier；两者使用相同 asm 约束。
普通 C++ 最终生成何种指令要看编译结果，不能先认定 `legacy` 等价于 `ca`。
Cache modifier 也不能保证特定的物理事务或并发请求数；若有收益，仍须区分编译结果和硬件机制。
这是基于 host 源依赖现象提出的可证伪候选，尚不是已确认的根因或优化收益。

生产设置 `SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=legacy|ca|cg`。
默认 `legacy`；首次解析后按进程固定，切换必须重启 decode 并重新捕获 graph。
启动时应在 decode rank 日志看到 `HiSparse DSv4 planned-copy load policy: ...`。
此设置覆盖 DSv4 的 planned-copy 调用，不给其他模型或融合 swap-in 路径自动应用新策略。

paired 基准显式指定每种策略，忽略该生产环境变量，以免误把候选当作基线。
三个 SM case 复用同一张卡上完全相同的 plan、counts、host 源及目的 buffer；顺序执行并同步。
新增 JIT cache key 包含策略，避免三种特化互相覆盖。

## 3. 先过新 CUDA 代码的正确性检查

实机首次执行在三种策略的 eager 校验上均失败，已定位为新增测试的布局错误：
测试把每页字节数写成 `38016`，生产 `kvcacheio.cuh` 实际是
`ceil(64 * 584 / 576) * 576 = 37440`（36864 value + 512 scale + 64 padding）。
kernel 按 37440 步进，测试分配和 oracle 按 38016 步进，第 p 页基址相差 `p * 576`。
`divmod(destination, 64)` 是正确的；这次失败不能作为第 65 个 miss 或 plan stride-96 有 bug 的证据。
测试输出的 tensor 行是缓存页，不是 request 行，跨页源和目的地址都受错误页宽影响。

CPU 按 kernel 的平坦地址公式模拟旧 fixture，复现第 1 行前三字节
`actual=[251,35,250]`、`expected=[181,223,108]`；改正页宽后完整字节一致。
修复复用了 paired 基准已有的正确布局常量，保留所有原始覆盖；没有修改生产 kernel/策略。
此前 paired/solo 基准和首轮 probe 的布局计算均正确，不需要因此重跑旧轮次。

当前开发环境没有 GPU/nvcc；本次相关测试为 **36 passed / 3 skipped**，
3 项跳过均为 CUDA 用例，修正后的 CUDA 执行仍需实机验证。
新增 CPU 回归直接构造同一测试 fixture：核对平坦物理地址与按页 oracle，
覆盖有效/空/padded request，并要求旧错误布局被拒绝。本项使用一张空闲的物理 GPU4，不启动模型。
Python 静态检查通过；全仓 CI 登记检查仍报 7 个既有文件缺少登记，已核实这些文件
与修改前 HEAD 完全相同，本次新增/修改测试的登记均已单独验证。

在已有诊断分支的仓库目录执行：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R6="$LAGOM_REPO/work/h2d-round6"
mkdir -p "$H2D_R6"

CUDA_VISIBLE_DEVICES=4 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hisparse_copy_load_policy.py \
  >"$H2D_R6/correctness-v2.txt" 2>&1
cat "$H2D_R6/correctness-v2.txt"
```

必须 **3 passed**，跳过不算通过。验证所有 value/scale 字节、未写区域及 page padding，
包含跨 page、非连续 plan 行、空 miss、padded request，以及 graph replay 之间改变 host 数据和计数。
若断言失败，新增 `Copy mismatch ...` 行会给出策略、eager/graph 阶段、counts、真实请求数、
页宽、plan stride、每 tensor 行错误字节数，以及前 8 个错误字节的实际/预期值。
反馈这一行即可；若编译或其他异常则给末尾约 30 行。在此停下，不继续性能实验。
无需清理 JIT 缓存；通过后直接继续 §4，不另跑前几轮对照。

## 4. 两次正式对照：先正序，再反序

只使用空闲物理 GPU4/5、NUMA1。保留 context=1、Portable=1、512 MiB 共享 host 池，
B=32、miss100 / miss1272、CTA4/1024。无需重跑 HBM 对照，也不扩大线程数矩阵。
每种方法两卡仍是 8 波 ABBA：`[0,1,1,0,0,1,1,0]`。

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --copy-load-policy-control \
  --profile --phases copy --epochs 2 \
  --output-dir "$H2D_R6/policy-forward" >"$H2D_R6/policy-forward.log" 2>&1
```

要求退出 0，`success=True profile_success=True`，correctness、graph replay 与
profile coverage 均为 **18/18**。通过后反转测量的 case/method 顺序，分配顺序和两卡 ABBA 顺序保持不变：

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --copy-load-policy-control --reverse-method-order \
  --profile --phases copy --epochs 2 \
  --output-dir "$H2D_R6/policy-reverse" >"$H2D_R6/policy-reverse.log" 2>&1
```

反序同样要求 18/18。这里 `--phases copy` 是无 profiler 的测量阶段；
profile 沿用已有的 compute 预热阶段，紧凑输出标为 `phase=profile_compute`。
只在同一次、同阶段、同卡、同 payload 内比较策略，不能把不同预热阶段的绝对时间相减。

先确认两次的 `sm` 基线仍复现，再比较 **每卡自己的候选/legacy 时间比**：

- 主指标是 miss100 的耗时下降。原正常 residency 的 1–4 MB 桶占约 89% copy 时间；
  miss1272 主要是压力对照，不能仅凭它获益宣布原工作负载优化成功。
- 看无 profile 的总 bytes / 总 event time，同时用同波 activity 验证下降是否在 kernel 内。
  要核查 CE、页位置和状态快照；若这次整个机器状态已变，不能归因于策略。
- 若 GPU5 的 miss100 在正反两次均下降约 5% 或更多、超过各波波动，且 GPU4 无明显退化，
  则进入生产验证。5% 是本轮筛选门槛，不是统计显著性证明；大载荷也须检查有无明显退化。
- 不要求 GPU4/GPU5 比值变成 1：两卡同时加速而比值不变，也可能是有用优化。

| 结果 | 下一步解释和动作 |
|---|---|
| `ca≈legacy`，`cg` 稳定更快 | 支持 cache modifier 对此 kernel 有影响；将 `cg` 作为生产候选，仍不直接宣称 ATS/TLB 或队列深度根因 |
| `ca` 与 `cg` 都比 `legacy` 快且彼此接近 | 优先考虑显式 PTX/codegen 改变；选择稳定收益者，不能专门归功于 `.cg` |
| 三者接近或候选退化 | 本候选未获支持，保留 `legacy`；下一步转向读取指令/流水化的具体干预，不继续重复同一对照 |
| 只有 23.77 MB 获益 | 不将其推广到正常 B32；先反馈结果 |
| 正反序收益方向不一致或校验失败 | 不进入服务 A/B；反馈紧凑结果定位 |

## 5. 微基准获益后，直接在原始服务验证

这部分等原来的 8 卡 PD 实验资源可用时执行，不占用其他人的 GPU0–3。
沿用已跑通的服务启动、路由和请求命令；见 [首轮服务和测量步骤](README_h2d_round1.md)。
不要为测试这个开关另换模型、开启强制 H2D，或同时调 batch / 线程数 / 亲和设置。
如果绑 NUMA，只能在两个策略组中同样应用，并记录实际 host pool 页位置。

在原 decode 启动命令前设置以下环境变量；`cg` 仅是示例，采用 §4 中真正获益的策略：

```bash
# 基线 decode 进程
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=legacy
export SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0
# 此处执行原 decode 启动命令；确认 rank 日志显示 legacy。

# 候选 decode 进程：停止自己的上一组 decode 后，用相同命令重启。
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=cg
export SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0
# 此处执行原 decode 启动命令；确认 rank 日志显示 cg。
```

端到端性能先关闭 probe 和 profiler，`force_h2d_each_step=false`。
采用四个测量块 **legacy → candidate → candidate → legacy**；每块重启自己的 decode，
执行同样的一次完整预热负载并丢弃，再执行三次相同负载，结果目录和标签各自独立。
prefill 的存活状态、warmup/cache 状态和 reset 方法四块保持一致。
仍使用固定 seed=0、64 个请求、最大并发 32、16384 输入 / 4096 输出：

```bash
# 每个测量块将 RUN_BLOCK 改为 legacy-1 / candidate-1 / candidate-2 / legacy-2。
export SHAREGPT_PATH=/home/jovyan/td69032/ShareGPT_V3_unfiltered_cleaned_split.json
RUN_BLOCK=legacy-1
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_R6/service" "$RUN_BLOCK-warmup"
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_R6/service" "$RUN_BLOCK-1"
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_R6/service" "$RUN_BLOCK-2"
bash scripts/pd_test/bench_h2d_round1.sh "$H2D_R6/service" "$RUN_BLOCK-3"
```

每次需 completed=64、input tokens=1048576、output tokens=262144。
反馈实际 output tokens/s、TPOT 及其重复波动、可用的 p50/p95、稳态 accept len/rate。
注意服务启动参数、CUDA Graph、实际 B/W 与物理编号也应对齐；accept 改变会影响每 token 成本。
吞吐未超过运行波动，就不能声称端到端收益，即使 copy kernel 明显更快。

随后单独做 legacy/candidate 各一次真实 decode profile：probe=16384，MAX_BATCH=1024，
正常 residency，在稳态实际 B=32 采集 64 scheduler steps，使用首轮 collector/analyzer。
报告每 rank 的 bytes、copy activity 总时间、完整 verify 迭代数、B/W、accept，
并按相近 payload 比较；各组真实 miss 不保证完全相同，不能只比时间总和。
从原始 trace 检查 copy 的下降是否缩短 verify 临界路径，或被其他 rank/计算/等待遮住。
这两次 profile 的 tokens/s 不进入前面的性能结论。

端到端验收需要同时闭合：**输出/接受行为正常 → 相近真实载荷的 H2D 更快 →
临界路径缩短 → 原始负载吞吐或 TPOT 的重复改善**。不能把连续 CE 的 55 GB/s
视作此随机 gather 必须达到的保证，也不能把两卡微基准的倍率直接换算为 8 卡吞吐收益。

## 6. 本次只需反馈

先完成 §3–4 即可；§5 已列出后续验收方法，不需要为此等待 8 卡资源再反馈。
只给 `correctness-v2.txt` 的结果和两份 `policy-*/pair-compact.txt`，不上传 trace/大日志。
若只能文字，给两次的 correctness/coverage，以及两档 payload、两张卡各自
`legacy/ca/cg` 的无 profile event 均值、profile activity 均值和 CE 是否稳定。
