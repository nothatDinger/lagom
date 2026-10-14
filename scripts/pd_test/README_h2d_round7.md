# 第七轮：保持同一地址集合，检验访问顺序

分支仍为 `perf/hisparse-h2d-diagnosis`，基于 `hash_verify`。本轮只用空闲物理 GPU4/5、
NUMA1、原生 venv；生产默认仍为 legacy。先 smoke，再一个正式对照，不重跑 8 卡服务。

## 1. 本轮依据与问题

第六轮 B32/W6 的 21 层 × 55 周期结果已闭合：ca 单位字节 copy 时间改善仅约
0.07/0.45/0.39/0.47%，主导 1–4 MB 桶略慢；12 次服务测量也没有可见收益。
ca 不晋升默认，本轮不再扫描 cache modifier。

仍待解释的现象是同一 host pool、单进程顺序双卡测量中，host SM 在 GPU5 稳定慢约 34–36%，
而连续 CE 和同 kernel 的本卡显存源基本对齐。差距在 kernel activity 内部。
已有 CTA=4/8/16/32/64 扫描没有显示增加 CTA 获益；这不等于排除了所有读请求并行度因素。
服务的慢 rank 现为约 22.9 GB/s、GPU4 约 31 GB/s。
筛选后的 1–4 MB 桶约占 copy 时间 71%，4 MB 以上合计约占 22%，保留小/大两档对照。

代码证据：`copy_cache_planned_kernel` 按 warp round-robin 遍历展开的 miss 列表；
`transfer_item` 以 64-bit lane load 读取 576-byte value 和 8-byte scale。
源和目的均是绝对 token-slot 索引，copy 不依赖目的属于当前 request。
本路径由 SM 直接读取注册的 host 地址，没有 CPU copy-worker 队列。
早期“线程少/队列浅”若指 CPU copy-worker，不能解释这个已隔离的 copy kernel 差距。
本轮检验：**同一批源地址只改变遍历次序，是否能明显改变 host SM 成本及两卡差距？**

## 2. 对照到底改变什么

新开关 `--source-order-control` 在既有 paired 工具中加入两种方法：

| 方法 | 源存储 | plan 顺序 |
|---|---|---|
| sm | 同一注册 host pool | 原随机顺序 |
| sm_ordered | 同一注册 host pool | 每个 bank 按源 slot 全局升序 |
| sm_hbm | 本卡显存源副本 | 原随机顺序 |
| sm_hbm_ordered | 同一显存源副本 | 相同升序 plan |
| ce | 同一 host pool | 连续 copy 参照 |

每个 bank 的所有 `(src,dst)` 对一起重排；源地址集合、目的地址集合、搬运字节、
每行 miss 数、CTA4/1024、legacy kernel、host pool 以及目的 buffer 均保持一致。
全局重排可能把一对搬到另一 request 行，但只使用其绝对目的 slot，最终内容不变。
因为 bank 的排列各异，**随机和 ordered 两组**都使用相同 shape/stride/分配尺寸的
三维目的 plan，避免只让 ordered 承担额外 plan 工作集；随机组只是将原目的表逐 bank 复制。
传给 kernel 的仍是该 bank 的二维 stride-3072 行视图。
其 HBM 版本分别直接复用对应 host 组的 GPU plan、counts 和目的 buffer。

CPU 构造后，用目的地址独立检验**所有 bank**的完整源到目的映射不变，结果记入 `ORDERED_PLANS`。
GPU eager 的首/尾 bank 和完整 graph replay 后，始终使用**原随机 plan**作为 CPU 字节 oracle，
同时验证全部未选中 slot 与 page padding 仍为零。不是拿新 plan 自己作预期结果。
`UNTOUCHED_SM checked=16/16` 是此配置的额外门禁。

排序与 plan 上传全部在计时外；本轮结果不能称为已含排序成本的生产收益。
源与目的必须一起重排，所以目的写入次序也变了；HBM 对照用于观察共同 GPU 端影响。
升序相邻 slot 也不保证所有 warp 的实际请求时间严格升序。
即使 host 排序变快，仍可能涉及 cache、地址转换、事务合并/排队或它们的交互，
不能据此单独宣称 ATS/TLB 根因，更不能用 host−HBM 时间当 PCIe 耗时。

## 3. 先运行一次新路径 smoke

在现有实验 worktree 更新分支，再设置路径。共享 host 仍 Portable=1，注册 context=1，
与最近 paired 对照一致。确认物理 GPU4/5 空闲后运行：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R7="$LAGOM_REPO/work/h2d-round7"
mkdir -p "$H2D_R7"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --source-order-control --device-source-control \
  --batch 2 --misses 2 70 --plan-stride 128 --buffer-slots 128 \
  --source-mib 4 --max-banks 4 --bulk-ce-mib 1 \
  --warmup 1 --warm-ms 100 --epochs 1 --phases copy --profile \
  --output-dir "$H2D_R7/order-smoke" >"$H2D_R7/order-smoke.log" 2>&1
```

需要退出 0，`success=True profile_success=True`，correctness、graph replay 与 profile coverage
均 **22/22**，`UNTOUCHED_SM checked=16/16`，两档 ORDERED_PLANS 均 pairs_preserved=True。
正确映射仍应为 GPU4/5、host VMA 在 node1。失败时停在这里，反馈 compact 的错误/门禁行，
不要继续正式测量。smoke 工作集很小，带宽没有性能判读价值。

## 4. smoke 全部通过后，只跑一个正式对照

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --source-order-control --device-source-control \
  --misses 100 1272 --phases copy --epochs 2 --profile \
  --output-dir "$H2D_R7/order-main" >"$H2D_R7/order-main.log" 2>&1
```

同样要求 22/22 和 16/16。默认 B32、plan stride3072（与生产 W6 缓冲一致）、512 MiB host pool，
两档每次载荷为 1.8688 / 23.771136 MB。两卡各有一个显存源副本。
不重叠 bank 轮换及两卡 8 波 ABBA 沿用已验证路径，排序与随机用的是同一地址集合。
因此比较时无需因 bytes 不同做校正，但仍需核对 NVML 状态和 raw wave 的方向一致性。
source footprint 相同且轮换载荷大于 L2，不证明没有 cache hit；报告始终是逻辑有效载荷率。

先看无 profile 的 mean event 时间，再用同波 profile activity 确认变化是否发生在 kernel 内部。
profile 的 HBM 两种方法每次 H2D bytes 都是 0，其 GB/s 是逻辑处理率。
原 host SM 基线应仍复现卡间差距；否则本轮暂不解释 ordered 对照。

| 结果 | 下一步含义 |
|---|---|
| host 排序明显更快，HBM 排序变化小，卡间差距收敛 | 支持 host 读取对访问顺序敏感；再反序确认，并从真实 plan 检查是否有类似可利用结构和排序成本 |
| host 更快，但两卡比例不变 | 顺序改善共同成本，主要卡间差距仍在；分别讨论两个现象 |
| host/HBM 的排序变化接近 | GPU 端排序后的访问也受影响，不能将改善专属归因于 host 地址转换 |
| host 排序基本无改善、卡间差距仍在 | 同地址集合的升序不能解决主要差距，下一步测 host SM 的请求延迟/并行度或 PCIe 读取配置 |
| 原 SM 基线不复现、状态不匹配或校验失败 | 保留失败证据，先处理混杂因素，不据此推进生产修改 |

此刻不自动安排反序/四卡/服务重测；正式结果有明显收益再定向确认。
排序即便获得约 10% 以上的改善，也只是值得继续调查的筛选信号，不是统计显著性或端到端收益。

## 5. 只需短报告

smoke 通过可用一行说明门禁，再给正式运行：

1. success/profile、correctness/graph/profile coverage、UNTOUCHED_SM、ORDERED_PLANS 是否全通过；
   UUID→物理 GPU4/5、host node1 和 Portable=1 是否正确。
2. 两档 × 两卡的 sm/sm_ordered、sm_hbm/sm_hbm_ordered：无 profile mean event µs、
   profile mean activity µs、gap；最好计算每卡 ordered/random 时间比。
3. CE 是否仍对齐，SM/MEM 时钟、Pstate、link/throttle 有无明显差异。

完整 JSON/trace 留在本机，不需上传。排序成本、真实 plan 收益和端到端收益本轮均未测。
