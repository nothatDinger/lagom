# 第五轮：反转注册 context，再替换 SM kernel 的源存储

分支仍是 `perf/hisparse-h2d-diagnosis`。只使用空闲的物理 GPU4/5、NUMA1，原生 venv 执行。
生产 kernel 和服务参数不变。本轮至多两项：先 A；仅当 A 仍复现原差距时再 B。

## 1. 第四轮确认了什么

旧配置 register_device=0（物理 GPU4），共享 host、冻结 miss plan、flags=1。
correctness 与 profile coverage 均为 10/10。时间单位 µs，每组 event/activity 来自同一个 profiled wave：

| 用例 | GPU4 event / activity | GPU5 event / activity | activity 时间比 GPU5/GPU4 |
|---|---:|---:|---:|
| SM 1.87 MB | 66.18 / 61.94 | 86.45 / 82.25 | 1.328 |
| SM 23.77 MB | 733.00 / 728.81 | 995.44 / 990.98 | 1.360 |
| 连续 H2D 1.87 MB | 41.74 / 36.26 | 41.90 / 36.57 | 1.009 |
| 连续 H2D 23.77 MB | 436.77 / 431.22 | 438.54 / 433.15 | 1.004 |
| bulk CE 64 MiB | 1216.51 / 1211.06 | 1227.12 / 1221.76 | 1.009 |

SM 两卡 activity 差约 20.31 / 262.17 µs，外围 gap 差约 -0.04 / +0.27 µs。
因此在当前实验里，事件外围间隙解释不了主要差距。额外耗时发生在 kernel activity 内部。
无 profile 的新进程基线也复现 1.326 / 1.356；它在 profiler 启用之前采集，
单凭这一点不能证明 profiler 无扰动，但 profiled activity 比值保持同形，构成额外佐证。

用户简表将 bulk activity 比值写成 1.006；原附件 1221.76 / 1211.06 = 1.0089，仍接近 1。
状态快照接近但不完全相同：小 SM 的 GPU4 为 1365/1485 MHz、GPU5 为 1485 MHz；
大 SM 为 1485/1500 MHz，memory 2619、P0、Gen5 x16、throttle 0x4。
这些状态不支持“慢卡明显低频”的简单解释，但不是全程状态证明。

还未拆开：miss plan 遍历、host read/地址转换、device scatter/store、GPU 内部排队/等待。
kernel activity 包含 stall，不等于有效指令一直在执行。CE 和 SM 的执行方式及访问形状同时不同，
仅凭 CE 接近不能排除所有 device-side 成本，更不能定位到某个 PCIe switch/ATS/TLB 机制。

## 2. A：仅改变注册调用的当前 CUDA context

现有 `--register-device` 在调用 `cudaHostRegister` 前设置 current device。
始终保持 `CUDA_VISIBLE_DEVICES=4,5`、logical devices 0/1、ABBA 顺序、flags=1、源池尺寸和 plan seed 不变。
这里“注册 context”只表示调用时的 context，不主张 Portable 内存由某卡独占或必然经过该卡转发。
也没有反转整个 CUDA 初始化顺序。

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R5="$LAGOM_REPO/work/h2d-round5"
mkdir -p "$H2D_R5"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --profile --phases copy --epochs 1 \
  --output-dir "$H2D_R5/register1" >"$H2D_R5/register1.log" 2>&1
```

要求退出 0，success/profile_success 都为 True，correctness/profile coverage 为 10/10。
本项使用已有采集路径，无需新 smoke。

- 若仍是 GPU4 快、两档 SM activity 比值仍约 1.3–1.36、CE 接近 1，则继续 B。
- 若差距收敛、翻转或发生其他明显变化，先停在 A 并反馈。后续会做原 context 的回测以区分 context 与运行漂移。
  每个新进程重新分配 host 内存，不能将单次翻转直接称为 context 因果证明。
- 如果校验失败，反馈错误末尾和已有 compact，先不跑 B。

## 3. B：同 kernel 的本卡显存源对照

`--device-source-control` 保留原 `sm` 和 `ce`，另增加 `sm_hbm`。
每卡在计时前创建一份约 512 MiB 的源池，内容与 host 完全一致；两档 miss 共用该卡的源池。
它通过同一个 `copy_cache_planned_mla`、同一编译配置 CTA4/1024 运行，直接复用对应 host-SM case 的
GPU miss plan、counts 和目的 buffer 分配；不是只保持相同尺寸，plan/目的指针也相同。
只将源参数指向测量卡自身的 CUDA tensor；不走 peer GPU。两种 graph 顺序执行并在切换前同步。
源分配地址和缓存/地址转换行为随来源变化，不能视作只有 PCIe latency 一项变化。

源池初始化的 H2D 在计时与 profiler 之外完成。`sm_hbm` 的每次测量 H2D bytes 为 0，
输出 GB/s 是逻辑 payload 处理率，不是 H2D 带宽，也不是总显存读写带宽；局部 cache 可能参与。
不得用 `T(sm)-T(sm_hbm)` 直接当作 PCIe 耗时，因为两条路径的 cache、重叠和瓶颈均可能改变。

新源路径仍在所有计时之前检查首尾 bank 的全部 value/scale 字节，随后清零输出、graph replay、检查最后 bank。
验证 oracle 始终是 CPU 上的原始 host 数据。profile 要求覆盖所有 14 个 scope，不把新增 HBM kernel 当作 CE。
本地无 GPU，新增源路径需要此次实机的内置校验；任何检查失败会退出。

```bash
CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --register-device 1 --device-source-control \
  --profile --phases copy --epochs 1 \
  --output-dir "$H2D_R5/device-source" >"$H2D_R5/device-source.log" 2>&1
```

要求 success/profile_success 都为 True，correctness/profile coverage 为 **14/14**。
相较 A，每张卡增加约 512 MiB 源池；plan 和输出 buffer 复用，不加载模型。
首先确认 B 自己的 host `sm` 仍复现差距，才能解释它与 `sm_hbm` 的对照。

| B 的结果 | 支持的方向 |
|---|---|
| host SM 仍约 1.36，HBM SM 接近 1 | 差距依赖 host-backed source 路径；继续区分随机源访问、地址转换、事务粒度和读请求并行度 |
| host 与 HBM SM 都明显慢在 GPU5 | 优先查共同的 GPU 端执行、plan、cache/store 或设备状态；不能只归因于 PCIe host read |
| host SM 在 B 中不再复现 | 先处理新进程/分配/显存控制池带来的扰动；不能用 HBM 结果排除原瓶颈 |
| A 的差距已翻转/收敛 | 停在 A；先回测注册 context，不扩大机制扫描 |

即使 HBM 对齐，也没有单独证明是 IOMMU/TLB 或物理路径延迟；下一步才会区分连续与随机源、事务形状和 kernel stall。

## 4. 只反馈紧凑结果

A 反馈 `register1/pair-compact.txt`；若执行 B，再给 `device-source/pair-compact.txt`。
不需要 JSON/trace。若只方便短文字，请给：

1. A/B 的 correctness、profile coverage、物理编号和注册 context。
2. 两档 payload，各卡 `sm` / `sm_hbm`（B 才有）的 activity µs 与 event gap，以及 CE 是否仍对齐。
3. 每组无 profile SM 均值/比值，以及时钟、Pstate、PCIe/throttle 是否异常。

注意：摘要的 `first/second rate` 是 GPU4/GPU5 带宽比；相同 payload 下等于 GPU5/GPU4 耗时比。
