# 第六轮：检验源读取 cache policy，并准备回到原始 decode

目标仍是 `hash_verify`、HiSparse + DSPARK、ShareGPT 派生 16K 输入 / 4K 输出、B=32
在 8×H100 上的实际 decode H2D 和端到端吞吐。本轮提交第一个默认关闭的生产优化候选，
用既有 paired 工具检验；微基准通过后，同一个开关直接进入原始工作负载。
分支仍为 `perf/hisparse-h2d-diagnosis`，原生 venv，不要求 Docker。

**当前进度：§3 已 3 passed；§4 正反序均 correctness/profile coverage 18/18；
§5 的 12 次服务测量已完成，未观察到 ca 的服务收益，继续保留 legacy。
§5.2 的补采结果已收到：四 rank 的全窗口单位字节 copy 时间改善约 0–1.34%，
没有复现微基准约 5% 的量级。下一步只执行 §5.3，从已有分析 JSON 筛选 B32/W6，
无需重新占用 GPU，也不重跑正确性、微基准或吞吐测量。**

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
这是基于 host 源依赖现象提出的可证伪候选；目前已观察到 §4.1 的微基准收益，
仍未确认主要瓶颈根因或真实服务收益。

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
3 项跳过均为 CUDA 用例。后续用户实机反馈修正后的 CUDA 用例 **3 passed in 78.69s**，
三个策略的 eager/graph replay、value/scale/padding 校验均通过。
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
| `ca` 更快、`cg` 更慢（本次实际结果） | 选择 `ca` 进入真实服务；两个显式读取策略的差异会影响性能，`ca` 相对 legacy 的具体获益机制仍需编译产物/硬件证据 |
| 两种候选均无稳定改善 | 本候选未获支持，保留 `legacy`；下一步转向读取指令/流水化的具体干预，不继续重复同一对照 |
| 只有 23.77 MB 获益 | 不将其推广到正常 B32；先反馈结果 |
| 正反序收益方向不一致或校验失败 | 不进入服务 A/B；反馈紧凑结果定位 |

### 4.1 已收到的正反序结果

附件 `policy-forward.txt` 和 `policy-reserved.txt`（后者 config 的
`reverse_method_order=True`，确为反序）。两次均 success/profile_success=True，
correctness/graph replay/profile coverage 均 18/18，无 profile errors。
共享 host 池在 node1、Portable=1、physical GPU4/5；两次分别是独立进程，
每个进程内每卡每方法有四波重复，不将这些波或两张卡当作完全独立的机器实验。

以下是 **无 profile 的 mean event µs**，每格为 legacy → ca（时间减少百分比）：

| 载荷 / 卡 | 正序 | 反序 |
|---|---:|---:|
| 1.87 MB / GPU4 | 64.82 → 62.75（3.19%） | 64.79 → 62.77（3.12%） |
| 1.87 MB / GPU5 | 85.82 → 81.49（5.05%） | 85.95 → 81.73（4.91%） |
| 23.77 MB / GPU4 | 733.69 → 711.21（3.06%） | 732.51 → 711.36（2.89%） |
| 23.77 MB / GPU5 | 995.23 → 941.26（5.42%） | 994.93 → 939.86（5.54%） |

GPU5 小载荷的波均值：正序 legacy 85.67–85.97、ca 81.38–81.75 µs；
反序 legacy 85.77–86.27、ca 81.61–81.86 µs。收益超过已观察的波动，约 5% 的筛选条件满足。
CE 小载荷两卡约 39.3 µs、大载荷约 434–436 µs；bulk 约 55 GB/s，正反序接近。
无 profile 快照均 SM 1980 MHz、MEM 2619、P0、Gen5 x16、throttle=0。

对应 profile activity 的 ca 时间减少分别为 GPU4 小载荷 3.00%/4.04%、
GPU5 小载荷 5.18%/4.90%、GPU4 大载荷 2.99%/2.75%、GPU5 大载荷 5.34%/5.24%。
小 SM gap 约 4.1–4.4 µs，没有与收益同量级的 gap 减少；主要改善发生在 kernel activity 内。
原简报的大载荷 733.9/712.1、995.3/942.4 是 profiled **event**，不应标作 activity。
profile 个别时钟快照含 1725 MHz 或更低频，不宜写成全程固定在 1455–1515 MHz；
无 profile 结果也复现收益，因此不依赖这些 profiled 快照的解释。

`cg` 的无 profile 时间在小载荷退化约 5.0%–6.3%，大载荷约 6.5%–7.1%。
这不是旧判读表里“ca/cg 都快且彼此接近”的形态，也不能简单归为统一的 inline PTX 收益。
尚未读取实际 PTX/SASS，不能仅从时间推出 legacy 的 load 指令与 ca 必然不同；
更不能宣称是 L1 命中、ATS/TLB 或硬件队列深度导致。

ca 后的两卡无 profile 时间比，小载荷约 1.30、大载荷约 1.32，较 legacy 的约 1.32/1.36
略有收敛，主要差距仍在。这个候选没有解释或消除主要的 per-card penalty。

## 5. 在原始服务验证 legacy 与 ca

这部分等原来的 8 卡 PD 实验资源可用时执行，不占用其他人的 GPU0–3。
沿用已跑通的服务启动、路由和请求命令；见 [首轮服务和测量步骤](README_h2d_round1.md)。
不要为测试这个开关另换模型、开启强制 H2D，或同时调 batch / 线程数 / 亲和设置。
如果绑 NUMA，只能在两个策略组中同样应用，并记录实际 host pool 页位置。

本轮明确采用 **ca**。把下面环境变量应用到原 decode 进程，重启并重新捕获 graph；
确认四个 decode rank 的启动日志均显示本组策略，默认配置仍为 legacy。
生产参数和代码沿用已经测试的版本，不需要先做新的 kernel 调参或 SASS 采集。

```bash
# 基线 decode 进程
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=legacy
export SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0
# 此处执行原 decode 启动命令；确认 rank 日志显示 legacy。

# 候选 decode 进程：停止自己的上一组 decode 后，用相同命令重启。
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=ca
export SGLANG_HISPARSE_H2D_PROBE_CAPACITY=0
# 此处执行原 decode 启动命令；确认 rank 日志显示 ca。
```

端到端性能先关闭 probe 和 profiler，`force_h2d_each_step=false`。
采用四个测量块 **legacy-1 → ca-1 → ca-2 → legacy-2**；每块重启自己的 decode，
执行同样的一次完整预热负载并丢弃，再执行三次相同负载，结果目录和标签各自独立。
prefill 的存活状态、warmup/cache 状态和 reset 方法四块保持一致。
仍使用固定 seed=0、64 个请求、最大并发 32、16384 输入 / 4096 输出：

```bash
# 每个测量块将 RUN_BLOCK 改为 legacy-1 / ca-1 / ca-2 / legacy-2。
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
吞吐包含 prefill、排队和客户端开销，结合 TTFT 与 TPOT 判断，而非只看总 tokens/s。
若 copy 占迭代临界路径的比例为 f，copy 耗时减少 5% 对总时间的直接贡献约为 `f * 5%`；
例如 f=4% 时只是约 0.2% 的量级。这里是示例，不是当前服务实测占比或收益预测。

全部 12 次测量结束后，用以下只读命令提取短表，不上传 JSONL/大日志。
文件名显式排除 warmup 和 profile；标签不证明进程实际策略，仍需上面的四 rank 日志核对。
TPOT 的 p50/p95 是请求级 TPOT 分布，不是逐 token 的 ITL 分布。

```bash
"$LAGOM_PY" -S - "$H2D_R6/service" >"$H2D_R6/service-compact.txt" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
fields = ["output_throughput", "mean_tpot_ms", "median_tpot_ms", "p95_tpot_ms",
          "mean_ttft_ms", "duration"]
rows = []
for block in ("legacy-1", "ca-1", "ca-2", "legacy-2"):
    for repeat in (1, 2, 3):
        path = root / f"{block}-{repeat}.jsonl"
        records = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if len(records) != 1:
            raise SystemExit(f"{path.name}: expected exactly one benchmark result")
        result = records[0]
        for key, wanted in dict(completed=64, total_input_tokens=1048576,
                                total_output_tokens=262144, max_concurrency=32,
                                random_input_len=16384, random_output_len=4096).items():
            if result.get(key) != wanted:
                raise SystemExit(f"{path.name}: {key}={result.get(key)!r}, expected {wanted}")
        values = [result[key] for key in fields]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise SystemExit(f"{path.name}: invalid timing/throughput")
        rows.append(path.stem + " " + " ".join(f"{v:.6f}" for v in values))
print("All 12 workloads validated; warmup/profile excluded.")
print("run output_tok/s mean_TPOT_ms median_TPOT_ms p95_TPOT_ms mean_TTFT_ms duration_s")
print("\n".join(rows))
PY
```

另给每块稳态 B32 的 accept len/rate（可沿用 `summarize_h2d_report.py --decode-log ...`）。
PD 场景 JSONL 的 accept_length 可能为空或来自不同统计窗口，不用它替代实际 decode 稳态记录。

随后单独做 legacy/ca 各一次真实 decode profile：probe=16384，MAX_BATCH=1024，
正常 residency，在稳态实际 B=32 采集 64 scheduler steps，使用首轮 collector/analyzer。
报告每 rank 的 bytes、copy activity 总时间、完整 verify 迭代数、B/W、accept，
并按相近 payload 比较；各组真实 miss 不保证完全相同，不能只比时间总和。
尤其检查 physical GPU6/7：此前微基准只覆盖 GPU4/5，TP4 的临界路径可能由其他 rank 决定。
从原始 trace 检查 copy 的下降是否缩短 verify 临界路径，或被其他 rank/计算/等待遮住。
这两次 profile 的 tokens/s 不进入前面的性能结论。

端到端验收需要同时闭合：**输出/接受行为正常 → 相近真实载荷的 H2D 更快 →
临界路径缩短 → 原始负载吞吐或 TPOT 的重复改善**。不能把连续 CE 的 55 GB/s
视作此随机 gather 必须达到的保证，也不能把两卡微基准的倍率直接换算为 8 卡吞吐收益。

### 5.1 已收到的服务结果及结论边界

`server-compact.txt` 的 12 个 workload 均通过既定 token 数校验。保留全部样本：

| 指标 | legacy（6 次） | ca（6 次） | ca 相对变化 |
|---|---:|---:|---:|
| 各次 output tok/s 的均值 | 3707.949837 | 3673.321886 | -0.934% |
| 各次 mean TPOT 的均值（ms） | 7.478204 | 7.506187 | +0.374% |
| 各次 output tok/s 的中位数 | 3750.807229 | 3735.032809 | -0.421% |
| 各次 mean TPOT 的中位数（ms） | 7.479055 | 7.498935 | +0.266% |

这些是 run 级描述统计，不把四个测量块里的重复视为完全独立实验，也没有做等效性检验。
本轮没有可见的端到端改善，不将 ca 提升为默认生产策略。
两个低吞吐 repeat（ca-2-3、legacy-2-2）原样保留；仅凭 TTFT/TPOT 均值尚不能把原因
确定为排队或预热，更不能排除它们与其他条件的关联。中位数同样没有显示 ca 获益。

目前不能推出“copy 不在关键路径”。这条 verify copy 在代码中是同步路径，
copy 即使位于依赖链上，也可能因占比很小、其他 rank 限制或其他耗时变化而无法表现出可测吞吐改善。
当时更基础的缺口是：约 5% 的微基准收益是否在真实四 rank、实际 miss 形状和 host pool 下保留。
现已补采，其全窗口结果和剩余的 B32 可比性检查见 §5.3。
也不能把“不保留”直接命名为 PTX 交互抵消，必须先核对载荷、布局/放置、设备及编译策略。

本轮稳态 accept len 约 4.9–5.3、rate 0.77–0.88，两策略没有明显分化；
不同于早期 C 的约 5.6，换算每提交 token 的 copy 成本时应使用本次相应窗口的 acceptance。
最初两份 `/start_profile` 200 ACK 只证明请求已受理；服务随后约 1 秒退出，
没有 trace/probe 完整产物，因此不把这两次记为有效 profile。

### 5.2 仅补 profile：等待本次 capture 的产物完成

本节的补采已经完成，保留以下流程供复现；当前下一步是 §5.3 的离线筛选。

等原 8 卡资源空闲后，按原启动命令分别跑 legacy/ca 两个 profile block。
每种策略按同样流程预热一次，随后发起一次原 64 请求负载；不重复三次吞吐测量。
在启动各自 decode **之前**设置（第二个 block 将 legacy 改为 ca）：

```bash
export SGLANG_PROFILE_V2=0
export SGLANG_HISPARSE_DSV4_COPY_LOAD_POLICY=legacy
export SGLANG_HISPARSE_H2D_PROBE_CAPACITY=16384
export SGLANG_HISPARSE_H2D_PROBE_MAX_BATCH=1024
```

保持正常 residency、16K/4K、并发上限 32、原来的模型/DSPARK/Graph/NUMA 设置。
确认四个 rank 策略正确，在日志里实际 B=32/W=6 已稳态且仍有足够输出 token 时，
**以前台进程执行**下面的 collector；第二次使用全新的 `profile-only-ca-v2` 目录：

```bash
"$LAGOM_PY" scripts/pd_test/collect_h2d_profile.py \
  --decode-url http://127.0.0.1:30000 \
  --output-dir "$H2D_R6/profile-only-legacy-v2" --steps 64 \
  --wait --expected-ranks 4 --timeout-seconds 600
```

等待依据来自当前 V1 exporter 的执行顺序：每个 rank 先完成 torch trace 导出，
经过 rank 间 barrier，最后导出 `.h2d.json`，随后才打印 `Profiling done`。
`--wait` 按**本次 profile_id**匹配 TP0–3 的文件，要求每个 rank 有非空 trace、
可完整读取的 probe、匹配的 capture/PID/device、非空且无 overflow 的样本。
旧日志的四条 `Profiling done`、旧 capture、半写入 JSON 和只有 HTTP ACK 都不会算成功。
为避免每秒读取巨大 trace，等待阶段不解压 trace；完成后仍须运行现有 analyzer 验证 trace 内身份和配对。
此等待选项限当前 V1 torch、每 TP 一个设备的未合并 profile；不要用于 nsys 或多副本合并采集。

驱动的 `stop_all` 应放在 **collector 以 0 退出且本次 `capture-complete.json`
的 `artifacts_complete=true`、四 rank 齐全之后**，并等待本次压测进程结束。
不要因收到 ACK 或固定等待 1 秒就进入清理；也要检查 `finally`/EXIT trap 是否绕过这个条件。
完成标记会原子写入，里面列出本次 capture_id、每 rank 的 PID/device、trace/probe 文件及样本数。
它是产物导出的完成信号，`timing_analysis_validated=false` 明确表示尚未通过下游数据分析。

超时会非零退出并保存 `capture-incomplete.json`，其中指出缺失的 rank/文件；
不要将超时当作成功继续清理。保留服务和产物，先检查负载/日志，必要时续等：

```bash
"$LAGOM_PY" scripts/pd_test/collect_h2d_profile.py \
  --output-dir "$H2D_R6/profile-only-legacy-v2" \
  --wait-only --expected-ranks 4 --timeout-seconds 600
```

`--wait-only` 使用该目录已保存的请求，**不会再次 POST**。先前已经终止的失败进程
无法靠续等补出 trace，应在新目录开始此次重跑。客户端需能看到服务写入的同一路径。

完成后用首轮 analyzer 分别生成两组四 rank 的分析 JSON，再用现有紧凑工具汇总。
保留未过滤的 B/captured/W 分布；若出现尾部 B30/31，不与 B32 混算，保留相应筛选说明。
优先比较同 B/W、同层及相近 bytes 的 copy 时间和占主导的 1–4 MB 载荷桶。

| 补采结果 | 下一步依据 |
|---|---|
| 四 rank 的真实 copy 都稳定更快 | 微基准收益在服务中保留；从这些已有 trace 提取完整 verify 边界、其他 kernels/等待，判断占比与抵消因素 |
| GPU4/5 获益、GPU6/7 无改善或更慢 | 查 TP 慢 rank 的约束，不用两卡结果推广四 rank |
| 真实 copy 没有改善 | 查实际 miss 形状、host NUMA 放置、状态和读取策略；不以端到端负结果代替 kernel 结论 |
| 两组 B/W、载荷或放置不匹配 | 先按可比样本分析并明确限制，不直接用总 copy 时间之比归因 |

一对 profile 是方向性证据，不是新一轮吞吐统计。若需更细时间线，先从本次保存的
trace 离线提取，不先安排新的 GPU 重跑。

### 5.3 已收到的生产 copy 结果：仅离线筛选 B32/W6

用户补采简报给出的总量如下。MB/GB 均为十进制；从已四舍五入的 bytes/time
计算 `1 - (T_ca / bytes_ca) / (T_legacy / bytes_legacy)`，因此百分比只取近似值：

| rank / 物理 GPU | legacy MB / ms | ca MB / ms | ca 单位字节耗时下降 |
|---|---:|---:|---:|
| TP0 / GPU4 | 2840.2 / 91.75 | 2848.7 / 91.95 | 约 0.08% |
| TP1 / GPU5 | 2839.2 / 125.05 | 2849.3 / 124.21 | 约 1.02% |
| TP2 / GPU6 | 2839.9 / 125.53 | 2848.3 / 124.22 | 约 1.34% |
| TP3 / GPU7 | 2840.3 / 124.25 | 2847.8 / 123.82 | 约 0.61% |

生产 profile 的全窗口没有复现此前两卡微基准约 3%/5% 的加速，
因而不能继续沿用“生产 copy 已快 5%，只是端到端掩盖了收益”的前提。
这仍不能证明 copy 不在关键路径，也不能将差异直接归因于 PTX 交互或四 rank 竞争。
TP0 约 31 GB/s、其他 rank 约 23 GB/s 的主要差距仍在，ca 没有解决它。

两组各 1344 个样本，但只有 **1155 个是 B32/W6**；另有 B1/W6 的 168 个样本，
以及 legacy B12 / ca B14 的 21 个样本。完整层序列若通过验证，1155 对应 21 层 × 55 周期，
不能将全部 64 周期都称为 B32 稳态。总字节归一化并不能消除不同 batch/层载荷的混合影响。
稳态 acceptance 范围相近（legacy 4.71–5.12、ca 4.75–5.03），目前只有范围、没有
与这些 55 周期对齐的加权均值；不据此计算精确的每提交 token 成本或临界路径占比。

更新后的紧凑工具可直接筛选**已经保存的 analyzer JSON**，不用读取大 trace、启动服务或导入 torch。
每个目录仍只放同一策略的一次 capture 的四 rank 分析 JSON。
将下面两个目录替换成这次实际的 legacy/ca 分析目录，从仓库根目录运行：

```bash
python3 -S scripts/pd_test/summarize_h2d_report.py \
  --analysis-dir /path/to/analysis-legacy \
  --analysis-dir /path/to/analysis-ca \
  --batch-size 32 --verify-width 6 --max-layers 21 \
  --output-file work/h2d-round6/service-b32-compact.txt
```

工具先验证原分析总量，再筛选真实 batch（不是 captured batch），重新计算 bytes、GPU 时间、
recorder 开销和各层/桶统计。保留零载荷 kernel 耗时及原始 index；不会把被删去的间隔拼成连续周期。
配对模式不是 `serialized_recorders`、没有匹配样本或输入总量不自洽时会拒绝筛选。
原始文件不变，输出的 hash 仍指向完整输入；`FILTER retained=1155/1344` 标出选择范围。
层表为每个 rank 分别提供 GB/s、时间占比、平均 MB/copy 和 n，便于核对同层载荷。
只有在序列连续且完整时才显示 `21/55`；否则显示 `unverified`，不能只拿 1155/21 猜周期数。

比较每个 rank 的 B32 总体及主导 1–4 MB 桶，然后比较相同层的平均 payload 和单位字节耗时。
两份 capture 的周期并非一一匹配；平均 payload 接近也不证明具体地址/每请求 misses 分布相同。

- 若可比的 B32/主导层仍只有约 0–1% 改善，本轮停止推进 ca；后续回到 host-backed SM
  读取的主差距，针对真实 miss/地址分布和四 rank 并发条件设计更有区分力的对照。
- 若 B32/主导层恢复了约 5%，说明全窗口混合确实影响判断，再从已有 trace 分析
  verify 依赖链及慢 rank；仍不能仅凭 copy 时间推断端到端收益。
- 若载荷差异明显，标为不可比；不通过挑选少数有利层给 ca 下结论。

默认策略继续为 legacy。这里不追加 GPU 重跑，也不把一个 profile 窗口当作统计显著性证据。

## 6. 本次只需反馈

只需 §5.3 生成的短文本，不需要再发已收到的吞吐、微基准或 acceptance 范围：

1. 八行 B32 rank 汇总（bytes、copy ms、GB/s、rec%、layer/cycle）及对应 FILTER/pairing 行。
2. 两组的 21 层表和载荷桶表；每层每 rank 的平均 payload 已由工具列出。
3. 仅补充已有记录中发现的异常：四 rank 是否实际选中了预期 policy、
   两组 host_numa_snapshot 的 KV pool 放置是否不同。未核实的项目写“未核实”，无需补采。

无需上传 trace 或完整服务日志。完整 verify 时间/临界路径未提取时标为未测，
不能用每周期 copy 时长之和替代整次迭代，也不能用此推断 copy 不在临界路径。
