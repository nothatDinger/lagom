# 第四轮：确认大载荷的时间差是否位于 copy kernel 内

第四轮 activity 已确认主要差距位于 kernel 内；后续见 [第五轮：注册 context 与源存储对照](README_h2d_round5.md)。

分支：`perf/hisparse-h2d-diagnosis`。本轮复用已有 GPU 采集代码，只补充紧凑报告中的时间分解和状态。
不需要新 smoke，不修改生产 copy 参数。本轮只安排一次短 profile。

## 1. 已取得的证据

旧 smoke 的离线恢复成功：correctness 10/10，profile coverage 10/10。
每个同名 scope 都是一个 CPU `user_annotation` 加一个 GPU `gpu_user_annotation`，
示例两者 `External id=105`，确认原解析器把 GPU 投影误计为第二个 CPU scope。
这不是重复 replay。

用户正式 pair-main：`success=True`，correctness 10/10，`profile=False`，因此只有 event 时间。
两卡读取同一 pinned allocation、同一计划；注册 flags=1，注册 context 为 logical 0（物理 GPU4）。
配置 B32、miss100/1272、plan/buffer 3072、source 512 MiB、CTA4/1024 threads、两个 ABBA epoch。
共享 buffer 的 VMA 只报告 node1 页面。页数是 VMA 口径，buffer 并未覆盖整个 VMA；不按字节比例分摊。

| 方法/载荷 | copy GPU4 / GPU5 (µs) | compute GPU4 / GPU5 (µs) | GPU5/GPU4 时间比，copy / compute |
|---|---:|---:|---:|
| SM，1,868,800 B | 64.65 / 85.56 | 65.52 / 86.23 | 1.324 / 1.316 |
| SM，23,771,136 B | 733.00 / 995.04 | 733.26 / 995.23 | 1.357 / 1.357 |
| 连续 H2D，1,868,800 B | 39.37 / 39.53 | 39.69 / 39.87 | 1.004 / 1.005 |
| 连续 H2D，23,771,136 B | 433.92 / 435.57 | 434.26 / 436.33 | 1.004 / 1.005 |

- copy 阶段两卡的前后状态均为 SM 1980 MHz、memory 2619 MHz、P0、PCIe Gen5 x16、throttle=0x0。
  四波 SM 均值稳定，跨卡差距远大于波间波动。
- compute 阶段 SM 时钟快照多在约 1.4 GHz，throttle=0x4；SM copy 小载荷只慢约 0.8%–1.35%，
  大载荷只变约 0.02%–0.04%。这削弱“核心时钟差导致 35% 差距”的解释。
  前后快照不是 kernel 全程遥测，且计算预热同时改变 cache，不能据此宣布所有状态因素已排除。
- 小连续 H2D 原先约 38 µs 的额外时间没有复现；剩余 SM 差距从约 21 µs 增到 262 µs。
  两个载荷点的描述性直线分别得到约 7.62/7.96 µs 截距和约 32.77/24.08 GB/s 的斜率带宽。
  仅两个点不能验证线性模型，但当前差距更接近随搬运量增长，不能沿用统一额外 40 µs 的解释。
- 两条 sysfs 路径均是 endpoint 加三个 bridge/root-port 节点，均报 Gen5 x16。
  它们是不同 BDF 路径；现有拓扑没有支持 GPU5 “多一个 switch hop”。
  这也不能证明两个 root port 的内部路由、read latency 或事务配置完全相同。
- 独立进程的 solo 与本轮同时改变了 allocation、注册 flags、context 和进程组织。
  小 CE 差距消失不能单独归功于共享内存，也不能将剩余差距直接命名为物理 GPU 固有上限。

## 2. 已对应上的代码事实

- `python/sglang/kernels/jit/csrc/hisparse.cuh` 的 `copy_cache_planned_kernel`：
  固定 grid、warp 轮转遍历 miss plan。CTA4 × 1024 threads = 128 个 warp，不是 4 个 CPU 搬运线程。
- DSv4 路径经 `copy_miss_item` 调到
  `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/kvcacheio.cuh::transfer_item`。
  一 warp 处理一条 miss：两组 value load 共 512 B，尾部再处理 64 B value 和位于页内另一段的 8 B scale，
  然后写入 device buffer。源索引是 gather；它不同于连续 H2D 的访问形状。
- 多 lane 读同一 scale 地址不等于 PCIe 重复搬运同样次数；实际事务合并/放大必须测量，不能从 lane 数推算。
- 测试中的 `capture_wave` 用 graph 内真实 event 节点包围每次 copy。
  得到的是 event 区间，可能包含活动前后等待；生产 copy kernel 的 GPU activity 需要 profiler 单独取出。

代码说明了潜在的 host read、地址转换、memory-level parallelism 与 device scatter 成本；
尚未证明哪一项造成当前差距。55 GB/s 是连续 CE 对照，不能视作该 gather kernel 必然可达的有效载荷带宽。

## 3. 只补一次短 profile

在原实验 worktree 更新；GPU4/5 空闲时运行，仍不使用 GPU0–3，不启动模型：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_REPO="$PWD"
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
export PYTHONPATH="$LAGOM_REPO/python${PYTHONPATH:+:$PYTHONPATH}"
export H2D_R4="$LAGOM_REPO/work/h2d-round4"
mkdir -p "$H2D_R4"

CUDA_VISIBLE_DEVICES=4,5 numactl --cpunodebind=1 --membind=1 \
  "$LAGOM_PY" scripts/pd_test/bench_h2d_pair.py \
  --profile --phases copy --epochs 1 \
  --output-dir "$H2D_R4/pair-profile" >"$H2D_R4/pair-profile.log" 2>&1
```

`--phases copy --epochs 1` 只将无 profiler 的对照缩到一轮 ABBA，验证新进程仍复现 SM 差距。
`--profile` 随后执行已有的 10 个 `profile_compute` scope。它仍有独立计算预热，
不因 `--phases copy` 变成 copy-only 预热；输出显式标明 phase 和该波的 NVML 前后状态。
载荷、banks、源池、目的布局、注册 owner 均保持原配置。一次新进程会重新分配 host 内存，
此轮不声称与上次使用相同物理页；每次进程内两张卡仍共享一份分配。

必须退出 0，`success=True profile_success=True`，correctness 10/10，profile coverage 10/10。
失败时只反馈错误末尾约 30 行和已有紧凑报告，保留 trace，不继续其他配置。

## 4. 最少反馈内容和判读

可直接给 `pair-profile/pair-compact.txt`，或提取以下约二十行：

```bash
rg '^(success=|CORRECTNESS |PROFILE_COVERAGE |PROFILE )' \
  "$H2D_R4/pair-profile/pair-compact.txt"
rg -A3 '^copy B32-miss(100|1272) sm$' \
  "$H2D_R4/pair-profile/pair-compact.txt"
```

PROFILE 行同时给出每次 copy 的 `mean_event/activity_us`、`mean_gap_us`、两种 GB/s、phase 和状态。
两种时间来自同一个 profiled wave，差值有意义；不要从旧无 profile 的 event 时间减去新 profile 的 kernel 时间。
保留极小的负 gap（可能在计时分辨率内），不截断成零。profiler 自身可能扰动执行，绝对吞吐仍以无 profile 为基线。

| 观察 | 下一步方向 |
|---|---|
| profiled SM activity 的跨卡差距仍约 1.3–1.36，event/activity 接近；CE 接近 | 进入 kernel 访问机制对照，并先拆开注册 context 与物理卡身份的关联 |
| activity 跨卡接近，差异主要在 event 减 activity 的间隙 | 排查 graph/event 调度、活动之间等待；不能把这部分叫 host read 带宽损失 |
| profiler 后差距明显改变、或新进程的无 profile 基线不复现 | 先检查 profiler/状态/分配干扰，不直接进入硬件根因结论 |

当前 register_device=0 恰好对应快卡 GPU4，这个关联还没有反转验证。
若下一步需要切换注册 context，会明确控制 CUDA 可见顺序、执行顺序和注册 owner；本轮暂不追加运行矩阵。
