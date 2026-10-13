# 第六轮：检验源读取 cache policy，并准备回到原始 decode

目标仍是 `hash_verify`、HiSparse + DSPARK、ShareGPT 派生 16K 输入 / 4K 输出、B=32
在 8×H100 上的实际 decode H2D 和端到端吞吐。本轮提交第一个默认关闭的生产优化候选，
用既有 paired 工具检验；微基准通过后，同一个开关直接进入原始工作负载。
分支仍为 `perf/hisparse-h2d-diagnosis`，原生 venv，不要求 Docker。

**当前进度：实机 §3 已 3 passed；§4 正反序均 correctness/profile coverage 18/18。
选择 `ca` 进入 §5 的真实服务对照，`cg` 不进入本次服务验证。无需重跑 §3–4。**

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

## 6. 本次只需反馈

§3–4 已完成，不需要再反馈旧数据。下一次优先给 §5 的 `service-compact.txt`，
每块稳态 accept len/rate，以及四 rank 策略、实际 B/W、probe=0 和无 profiler 的确认。
待原 8 卡资源可用再执行服务对照；仅有两卡时保留现有结果，无需追加微基准来等待。

单独 profile 后，反馈 legacy/ca 两组各 rank 的紧凑 H2D 表，包含 bytes、copy 时间、
GB/s、B/captured/W、完整 layer/cycle 数和 recorder 开销；可沿用
`summarize_h2d_report.py --analysis-dir ... --max-layers 21`。不上传 trace 或完整服务日志。
如果暂不能提取完整 verify 临界路径边界，明确标为未测，不用 copy 时长之和替代完整迭代。
