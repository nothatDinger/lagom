# 第九轮：真实 584-byte gather 的 lane 分配与向量 load

分支 `perf/hisparse-h2d-diagnosis`，基于 hash_verify。目标仍是正常驻留 B32/W6 的
decode H2D 和服务吞吐；物理 GPU4/5、NUMA1、原生 venv。
§3 与 §4 首个正式块均已返回结果。当前不推进反序或服务验证；见 §5 的判读。
下一步为 [第十轮 NCU 工具兼容性检查](README_h2d_round10.md)，不是再次运行策略扫描。

## 1. 依据与待检验问题

第八轮完整 host 曲线（16 个配置）卡间比值为约 1.0014–1.0249；同 N 改变 CTA
分布，变化约 −3.1% 至 +2.5%。它没有复现生产 gather 的 34–36% 差距。
不过 probe 每 warp 只有 lane0、每步一条依赖性 8-byte `.cv` load，生产则读取
584-byte KV 并写入显存。不能把这次负面结果当作生产请求并行度充足或 TLB 无关的证明。

这轮直接使用 `copy_cache_planned_kernel`、真实 C4 分页布局和相同 GPU miss plan。
代码位置：`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh`。
`transfer_item` 的前 512B 原本由每 lane 两次 64-bit load 完成；尾部 64B 与 scale 8B
共用第三次 load，scale 由 lane8–31 读取同一地址。广播不等于 24 次 PCIe 请求。

问题是：**在相同有效载荷和地址集合下，改变 lane 内 word 分配、再合并成向量 load，
能否改变 host gather 成本及两卡差距？** 这可能涉及指令发射、访问合并、cache 或请求排队；
单次收益不直接定位其中任何一个，也不预设 GPU5 的差距必须消失。

## 2. 四组写法，两种源

每个 64-bit word 为 8B。下表只干预前 512B（word0–63）：

| policy / host 方法名 | lane L 读取的两个 word | 源码 load |
|---|---|---|
| legacy / sm | L、L+32 | 原 C++ scalar |
| ca / sm_ca | L、L+32 | 两次显式 `.ca.b64` |
| ca_pair / sm_ca_pair | 2L、2L+1 | 两次显式 `.ca.b64` |
| ca_v2 / sm_ca_v2 | 2L、2L+1 | 一次显式 `.ca.v2.b64` |

四者各有本卡显存源对照：`sm_hbm`、`sm_hbm_ca`、`sm_hbm_ca_pair`、`sm_hbm_ca_v2`。
HBM 的每次 H2D bytes 为 0，其 GB/s 仅为逻辑载荷处理率。CE 及 bulk CE 沿用原对照。

- `ca_pair/ca` 比较 lane 内 word 分配，**相应 store 的 lane 分工也改变**。
  ca 每条前缀 load 覆盖连续 256B；ca_pair 每条覆盖跨 512B 的隔 word 集合，
  两条触及的地址范围重叠，所以它可能退化，并非预设的优化。
- `ca_v2/ca_pair` 保持相同 lane→地址及 C++ store 写法，比较 scalar/vector load。
  PTX 合并不保证 SASS 合并；寄存器数、编译调度和 store 产物也可能随之变化，需检查。
- `ca_v2/legacy` 才是最终候选与当前基线的性能比较。

相同内容：host pool、GPU plan/counts、目的 allocation、584B 载荷、CTA4/1024、
尾部/scale 的 lane 分配。新组均为 `.ca`，不混入 `.cg` 或 `.cv`。
不做排序、CPU packing、CE staging，也不增加 CTA；所有新组共享 legacy 的 GPU plan 和目的 buffer。
原来的 legacy/ca/cg 源码路径语义保持；需基线验证，不声称编译后二进制必然逐字节相同。

`ca_pair/ca_v2` 仅可显式传入实验 API，服务环境变量仍只接受 legacy/ca/cg。
向量读取要求 16B 对齐：mmap/CUDA allocation 的源基址先检查，page stride37440、
value stride576 和 lane offset16 都是 16B 的倍数；scale 仍是标量读取。

## 3. 正确性、paired smoke 与 SASS（已完成，命令留作复现）

2026-10-14 实机反馈：单测 **10 passed / 0 skipped**；smoke 的 correctness、graph、
profile **38/38**，未触碰字节检查 **32/32**，身份 GPU4/5、host node1、Portable=1。
实际加载库离线提取 **4/4**。两处 harness 兼容性修复已核对：Git ownership 检查失败
只记录 `git=None` 与 `git_error`，七个源文件 SHA 保留；cuobjdump 不传其不支持的
`--demangle`，解析器直接识别 mangled kernel 名。没有修改 ownership/safe.directory 设置。

实测 SASS 摘要（四者另有 `LDG.E.CONSTANT ×4`、`STG.E.64 ×3`）：

| policy | 其余 load opcode | registers | local / shared / stack |
|---|---|---:|---|
| legacy | `LDG.E.64.CONSTANT ×3` | 34 | 0 / 0 / 0 |
| ca | `LDG.E.64.STRONG.SM ×3` | 34 | 0 / 0 / 0 |
| ca_pair | `LDG.E.64.STRONG.SM ×3` | 32 | 0 / 0 / 0 |
| ca_v2 | `LDG.E.128.STRONG.SM ×1` + `LDG.E.64.STRONG.SM ×1` | 32 | 0 / 0 / 0 |

这确认 `ca_v2` 的宽 load 落到了 SASS，且 `ca_pair/ca_v2` 寄存器总数相同。
128-bit 是每个参与 lane 的 16B load，并非 128B PCIe 事务，也不证明请求数减半。
opcode 直方图不包含完整指令顺序、地址计算或依赖；不能仅据此称 legacy/ca
“只有 cache modifier 不同”，或称 `ca_pair/ca_v2` 除 load 外的二进制完全相同。
这些门禁支持测量候选；正式载荷结果另见 §5，不能用 smoke 时延判断收益。

在实验 worktree 更新并设置环境。GPU 单测会覆盖五种 policy × host/HBM，共十项；
包含 eager、动态 graph replay、非连续 plan stride、跨页目的、零 miss、padded request、
修改源内容后的 replay，以及全部输出字节/未触碰 slot/page padding。

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R9="$LAGOM_REPO/work/h2d-round9"
mkdir -p "$H2D_R9"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" -m pytest -q \
  test/registered/kernels/ops/kvcache/test_hisparse_copy_load_policy.py \
  >"$H2D_R9/correctness.log" 2>&1
```

需要 **10 passed，0 failed，0 skipped**。若失败，停止并返回断言/编译错误尾部。
本地 CPU 测试不能替代这个门禁。

通过后运行 paired smoke。复用已验证的每设备独立 capture stream，新增方法使用同一采集/解析流程：

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --copy-vector-control --device-source-control \
  --batch 2 --misses 2 70 --plan-stride 128 --buffer-slots 128 \
  --source-mib 4 --max-banks 4 --bulk-ce-mib 1 \
  --warmup 1 --warm-ms 100 --epochs 1 --phases compute --profile \
  --output-dir "$H2D_R9/vector-smoke" >"$H2D_R9/vector-smoke.log" 2>&1
```

门禁：退出 0，`success=True profile_success=True`，CORRECTNESS、graph replay、
PROFILE_COVERAGE 均 **38/38**，`UNTOUCHED_SM checked=32/32`，errors 为空。
来源是两档 × 九种方法 × 两卡，加两条 bulk CE；没有读链工具的 CAPTURE_DEVICE_CHECK 字段。
核对 UUID 为物理 GPU4/5、host node1、Portable=1、HBM 行 `h2d_bytes_per_copy=0`。
smoke 只验证功能与解析，不用于判断收益。失败保留原文件，返回 compact 错误。

接着离线提取本次实际加载的四个 JIT 库。此步骤不重新执行 GPU kernel：

```bash
"$LAGOM_PY" scripts/pd_test/collect_h2d_codegen.py \
  --pair-json "$H2D_R9/vector-smoke/pair.json" \
  --output-dir "$H2D_R9/vector-smoke-codegen"
```

需要 CUDA toolkit 的 `cuobjdump`；若不在 PATH，显式加
`--cuobjdump /实际/CUDA/bin/cuobjdump`。缺工具或产物提取失败只反馈错误，无需重跑 GPU。
采集端从 `/proc/self/maps` 记录实际加载库的路径与 SHA，拒绝把“缓存里最新的 .so”当作证据；
离线脚本再次核对 SHA，输出完整 SASS、resource usage、静态 memory opcode 统计。
`LDG` 还可能读取 GPU plan，静态条数也不是动态 PCIe 请求数；需要结合地址计算识别 source load。
保留完整文件本地，仅返回 `codegen-compact.txt`（通常约十行）。

上述门禁与 codegen compact 已核对；完整 trace/SASS 留在实机，无需上传。

## 4. 首个正式对照块（已返回，命令留作复现）

保持已验证的 B32、miss100/1272、512 MiB pool、plan stride3072。
这轮无 profiler 主测选 `phase=compute`，每波先做相同计算预热；profile 也为 profile_compute。
两者仍是独立测量段，不能拿一个段的 event 减另一个段的 activity。
本次可直接使用实机上已通过 §3、含两处 harness 修复的版本运行；无需为了文档更新先同步
分支或重跑门禁。`vector-main` 必须是新目录。若使用新 shell，先恢复 §3 的四个环境变量。

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --copy-vector-control --device-source-control \
  --misses 100 1272 --phases compute --epochs 2 --profile \
  --output-dir "$H2D_R9/vector-main" >"$H2D_R9/vector-main.log" 2>&1
```

先只跑一个正式块。门禁仍为 38/38、32/32，legacy host 原卡间差距需复现，
CE/HBM 与状态检查有效。若这次 baseline 不再复现，先解释环境变化，不将新组用于归因。

先返回 miss100 的四种 policy × 两卡：无 profile mean event、同一 profiled wave 的
mean event/activity、四波均值，以及对应 HBM 和 CE 比值/状态。大载荷保留作负载趋势检查。
主要比较 `ca_pair/ca`、`ca_v2/ca_pair`、`ca_v2/legacy` 的同卡耗时比；不要只追求卡间比值归一。

| 结果 | 下一步 |
|---|---|
| ca_pair 已改善，ca_v2 无额外收益 | 优先检查 lane 地址分配、合并/cache 与 store 分工；不能归功于向量指令 |
| ca_v2 相对 ca_pair 改善，且 SASS 确有宽度变化 | 支持 load 组织/代码生成敏感；HBM 参照用于评估共同执行影响，不直接证明 Tag/TLB |
| host 与 HBM 均类似改善 | 可能有共同 GPU 端收益；不能称为专属 PCIe 改善 |
| 目标 miss100 无可重复改善或退化 | 停止候选，不靠大载荷或卡间比值为它晋升 |
| miss100 对慢卡 legacy 有约 10% 以上改善且跨波稳定 | 做方法反序确认，然后验证真实 decode 的单位字节 copy 时间和无 probe 吞吐 |

10% 是减少无效服务实验的筛选门槛，不是统计显著性证明。
回到服务时沿用第六轮经验：micro 收益必须在 B32/W6 过滤后的真实 copy 中出现，
再检查 TPOT/吞吐。不能重复把未进入服务的 5% micro 收益解释为已完成优化。

## 5. 正式块反馈与当前决定（2026-10-14）

用户报告采集版本 `6a5f23465`，物理 GPU4/5、共享 host pool node1、Portable=1。
主测 `success=True profile_success=True`，profile scope 均为一个 CPU annotation。
STATE 摘要两卡 SM 1455–1485、MEM 2619、P0、Gen5 x16、throttle 0x4、errors=0；
这不是每个 kernel 内状态恒定的证明。正式块已补充确认 correctness/payload/graph/profile
均为 **38/38**、未触碰字节检查 **32/32**、profile errors 为空；不依赖 smoke 计数代替。

miss100 的无 profiler event 是本轮性能主口径；profile activity 用于内核内耗时分析。
两段分别计算，所有百分比按用户提供的四舍五入均值推算：

| 同卡比较 | 无 profile event d0 / d1 | profile host activity d0 / d1 |
|---|---|---|
| ca / legacy | −3.54% / −4.22% | −2.75% / −7.29% |
| ca_pair / ca | −0.89% / −1.82% | −0.48% / −1.51% |
| ca_v2 / ca_pair | +0.22% / −0.17% | −1.50% / −0.52% |
| ca_v2 / ca | −0.67% / −1.99% | −1.98% / −2.03% |
| ca_v2 / legacy | −4.19% / −6.12% | −4.67% / −9.17% |

无 profile 的 legacy 为 66.18 / 86.12 µs，ca_v2 为 63.41 / 80.85 µs。
慢卡节省 5.27 µs/次；不能把另一段 activity 的 9.17% 当成无 profile 性能收益。
profile 的 event−activity gap 保持约 4.2 µs，只说明这段区间外开销相近；
它不证明 profiler、运行次序或状态对各次 kernel 内部时间的影响相同。
暂不计算显著性或宣称 1–2% 的差异稳定可重复。

d1 四个无 profile wave 的均值如下，每波 64 个样本：

| policy | 四波均值（µs） | 最大−最小（µs） |
|---|---|---:|
| legacy | 85.95 / 86.23 / 86.19 / 86.12 | 0.28 |
| ca | 82.27 / 82.98 / 82.66 / 82.06 | 0.92 |
| ca_pair | 80.84 / 80.53 / 80.53 / 82.08 | 1.55 |
| ca_v2 | 81.25 / 80.78 / 80.62 / 80.76 | 0.63 |

不能概括为“四波抖动均不超过 0.5 µs”。按波序号比较，ca_v2 前三波略慢于 ca_pair、
第四波较快；平均仅快 0.14 µs。不同方法依次测量，序号对应也不等于同时测量或独立重复实验。
约 2.0% 指 ca_v2 相对 ca，约 0.2% 才是相对 ca_pair；两种分母不可互换。

HBM 对照也发生变化：

| 同卡比较 | profile HBM activity d0 / d1 |
|---|---|
| ca / legacy | +3.25% / +2.63% |
| ca_pair / ca | +5.75% / +7.31% |
| ca_v2 / ca_pair | −6.10% / −6.56% |
| ca_v2 / legacy | +2.52% / +2.91% |

`ca_v2/ca_pair` 在 HBM 下明显下降，在 host 下变化很小；支持更宽 load 的 GPU 端收益
受到 host 读取限制，但没有独立定位限制发生在哪一级。HBM 的两卡比值接近 1 不等于
四种方法在 HBM 的耗时相同，不能概括为“通用 GPU 执行无影响”或相减得到 PCIe 耗时。
CE 两卡 activity 36.27 / 36.44 µs，沿用其连续传输参照身份。

代码与归因边界：

- `ca_pair/ca` 的 lane 重排只显示小幅变化；不足以解释或解决主要差距，不能推广为 lane 分配无影响。
- `ca_v2/ca_pair` 的向量化在目标 host 载荷没有明确的额外收益。大载荷用户给出的
  activity 比值为 0.9996 / 0.9970；未据此宣称其他大载荷指标已核验。
- `ca/legacy` 已包含本组大部分 host 改善，但只能归因到这两个**实现整体**的差异。
  SASS opcode 统计不提供完整指令调度、依赖、cache 命中率、物理事务或翻译计数。
  `.CONSTANT` 与 `.STRONG.SM` 的出现不能单独证明某个 cache modifier 是硬件根因。
- `transfer_item` 的 `tail_item` 同样调用 `load_item_word<LoadPolicy>`：legacy→ca
  还改变了尾部 64B value 与 8B scale 的 load 策略；仅 ca→ca_pair→ca_v2 保持这部分一致。
  因此“四组只改前 512B”不是准确的全组描述。

**当前不推进 ca_pair/ca_v2 的反序及服务实验，服务仍使用 legacy。**
慢卡无 profile 的 ca_v2/legacy 为约 −6.1%，相对已有 ca 仅约 −2.0%；未达到本轮约 10%
的筛选目标。第六轮 ca 的 micro 收益没有进入真实服务，不能仅凭本轮单个 profile 波再晋升。
这不是判定硬件根因已经找到，也不是判定 copy 不在端到端关键路径。

下一步先确认 Nsight Compute 的可用性，随后针对真实 legacy gather 采集访存与等待证据。
保持同一 host 分配、plan、目的 buffer，并配本卡 HBM 参照；先区分读流量放大与等量读取更慢，
再决定是否干预事务组织、并发或地址转换。计数器名称/权限需依实机工具确认，
暂不承诺某个 PCIe/TLB/队列指标可用，也不把 L1/L2 sector 数直接称作 PCIe TLP 数。
NCU replay/缓存/时钟策略会改变测量条件，届时单独核验；其 profile 时延不替代无 profile 基线。

所需旧数据已补齐，不需要继续提取第九轮文件。工具条件也已确认：`/usr/bin/ncu`
为 2022.4.1.0，`RmProfilingAdminOnly: 0`。该参数放开驱动的管理员限制，但不证明
此 NCU 版本已能采集当前 CUDA 13/JIT 产物；第十轮先验证兼容性，不修改驱动设置。
