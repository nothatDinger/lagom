# 第七轮：保持同一地址集合，检验访问顺序

分支仍为 `perf/hisparse-h2d-diagnosis`，基于 `hash_verify`。本轮只用空闲物理 GPU4/5、
NUMA1、原生 venv；生产默认仍为 legacy。先 smoke，再一个正式对照，不重跑 8 卡服务。

**当前进度：用户确认 smoke/main 的全部门禁通过，两列均取自同一个 profiled wave。
但简表的部分 event/activity 数值与当前校验条件不相容，先按 §6 只读原 JSON 核对。
按现有简表，1.87 MB 排序改善不足 1%，23.77 MB 有小幅改善，卡间差距没有收敛；
本轮不推进排序进入生产。后续 §7 是只读 PCIe 配置检查，不增加 GPU 重跑。**

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

## 6. 已收到的结果与口径核对

用户已从原 JSON 确认：smoke/main 均 success=True、profile_success=True，correctness、
graph replay、profile coverage 全部 22/22，errors=[]；UNTOUCHED_SM 16/16，
两档 ORDERED_PLANS pairs_preserved=True，Portable=1。
用户也确认两列均来自 `profile_waves/phase=profile_compute` 的同一条记录；
独立的无 profiler `waves/phase=copy` 只作基线。此前“可能跨阶段取数”的推测撤回，
不能当作已查明原因。

下面是直接用用户简表重算的 ordered/random，尚未核对原 JSON：

| 来源/载荷 | GPU4 event 比 | GPU5 event 比 | GPU4 activity 比 | GPU5 activity 比 |
|---|---:|---:|---:|---:|
| host 1.87 MB | 0.994 | 0.996 | 0.993 | 0.991 |
| host 23.77 MB | 0.937 | 0.967 | 0.943 | 0.968 |
| HBM 1.87 MB | 0.998 | 0.997 | 1.000 | 0.998 |
| HBM 23.77 MB | 1.003 | 1.003 | 1.030 | 0.991 |

例如大载荷 HBM activity 是 `280.46/272.23=1.030`、`276.08/278.63=0.991`，
不是简报末尾的 1.003/1.004。host 大载荷 activity 也应为 0.943/0.968，
而不是 0.946/0.964。

还有一个必须闭合的字段定义矛盾。当前 `b0e3c0f53` 的
`python/sglang/benchmark/hisparse_h2d_pair.py::activity_check` 中：

```python
active = sum(e["dur"] for e in copies)   # activity_gpu_us，总量
interval = sum(wave["copy_us"])         # event_gpu_us，总量
if active > interval + max(1, 0.5 * len(copies)):
    # valid=False，进而 profile_success=False
```

`copy_us` 来自 CUDA Event 的 `start.elapsed_time(end) * 1000`。
**event_gpu_us 不是 CPU 或 GPU user_annotation 的持续时间**；CPU 标注范围用于筛选该波的
GPU activity，External id 也不是 CUDA Event 的计时来源。
compact 的两个 mean 使用同一个分母 `activity_check.activities`。
当 N≥2 时，门禁最多容许 activity 平均每次超过 event 0.5 µs。
表中 GPU4 HBM 小载荷为 `31.36−30.20=1.16 µs`，大载荷为
`272.23−258.39=13.84 µs`，均无法在这版代码中同时得到 valid=True。
两位小数的舍入不足以解释差异。

因此，已报告的通过状态与简表数值尚不能同时复核；需要定位原字段、汇总或代码版本的差异。
目前证据不足以判断 profiler 或 CUDA 计时本身是否有问题。

按现有简表可保留的方向性观察（数值待原字段核对）是：

- 小载荷 host 排序改善不足 1%；主要卡间差距仍在。
- 大载荷 host 排序有约 3–6% 的改善，GPU4 收益更大。activity 的 GPU5/GPU4
  从约 1.358 变为 1.393；event 列从约 1.358 变为 1.400，均未收敛。
- 目前不把源排序作为生产候选继续推进：正常驻留目标载荷改善不足 1%，且实际排序成本尚未计入。
- **未排除地址转换开销**。排序保留了源页集合，首次访问和地址转换工作仍可能存在；
  排序收益小并不证明 TLB/ATS 无关，也不测量真实在途读请求数。

请直接运行以下纯标准库命令，读取已经存在的 `order-main/pair.json`。
仅输出 1 行文件/版本信息和 8 条大载荷原字段汇总；不导入 torch，不加载 trace，不重跑 GPU。
请原样反馈输出，不重新手算或转表。`current_git` 是提取时版本，不能替代采集时版本。

```bash
python3 - work/h2d-round7/order-main/pair.json <<'PY'
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

p = Path(sys.argv[1])
raw = p.read_bytes()
r = json.loads(raw)
helper = Path("python/sglang/benchmark/hisparse_h2d_pair.py")
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
print(json.dumps(dict(file=str(p.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
    current_git=git.stdout.strip(), current_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),
    success=r.get("success"), profile_success=r.get("profile_success"),
    coverage=r.get("profile_validation"))))
rows = [w for w in r["profile_waves"] if w["case"] == "B32-miss1272"
        and w["method"] in ("sm", "sm_ordered", "sm_hbm", "sm_hbm_ordered")]
for w in rows:
    a = w.get("activity_check", {})
    n, e, s, t = a.get("activities"), sum(w["copy_us"]), a.get("event_gpu_us"), a.get("activity_gpu_us")
    print(json.dumps(dict(scope=w.get("scope"), phase=w.get("phase"), valid=a.get("valid"),
        banks=w["banks"], event_count=len(w["copy_us"]), activities=n,
        event_sum_us=e, stored_event_us=s, activity_us=t,
        stored_event_matches_sum=math.isclose(e, s, rel_tol=1e-12, abs_tol=1e-6) if s is not None else None,
        means_us=[s/n, t/n] if n and s is not None and t is not None else None,
        current_timer_gate=t <= e + max(1, 0.5*n) if n and t is not None else None)))
if len(rows) != 8:
    print(json.dumps(dict(unexpected_rows=len(rows), expected_rows=8)))
PY
```

若路径不同，只替换输入文件路径。从仓库根目录执行，以便记录当前 helper 的哈希。
若这 8 行自洽，则修正简表并继续；若存储 valid=True 但 current_timer_gate=False，
再核对采集时实际加载的 helper 和离线重解析版本。完整 pair.json 与 trace 继续留在本机。

## 7. 下一步：读取 PCIe 请求配置，不占用 GPU 计算

已有 `pci_path()` 仅从 sysfs 读取 current/max link speed/width 和 numa_node，
没有检查 PCIe capabilities/control。CE 两卡约 55 GB/s、Gen5 x16 相同，并不直接证明
SM 发起小读请求时的配置、在途事务限制及延迟都相同。
先检查可见配置差异，再决定下一次延迟/并行度实验，避免继续盲扫 kernel 参数。

先核对当前物理编号、UUID 和 BDF：

```bash
nvidia-smi -i 4,5 --query-gpu=index,uuid,pci.bus_id --format=csv
```

此前 GPU4/GPU5 分别为 `GPU-dd86c4db-6f83-d828-ccd4-cc2e9a407702` / `0000:98:00.0`
及 `GPU-e178b6af-9dd7-1539-4498-a37c2db6d653` / `0000:b8:00.0`。
已保存的 sysfs 路径配对为：

| 路径位置 | GPU4 路径 | GPU5 路径 |
|---|---|---|
| GPU endpoint | 0000:98:00.0 | 0000:b8:00.0 |
| 上一级 | 0000:97:00.0 | 0000:b7:00.0 |
| 再上一级 | 0000:96:00.0 | 0000:b6:00.0 |
| 最上游已记录 PCI function | 0000:95:01.0 | 0000:b5:01.0 |

若当前 BDF 不同，先反馈当前身份，避免使用旧路径。BDF 相同时，运行以下只读命令：

```bash
mkdir -p work/h2d-round7/pcie
for bdf in 0000:98:00.0 0000:b8:00.0 \
           0000:97:00.0 0000:b7:00.0 \
           0000:96:00.0 0000:b6:00.0 \
           0000:95:01.0 0000:b5:01.0; do
  lspci -D -s "$bdf" -vv >"work/h2d-round7/pcie/$bdf.txt" 2>&1 || break
done
rg '^[0-9a-fA-F]{4}:|DevCap|DevCtl|MaxPayload|MaxReadReq|ExtTag|10BitTag|ATSCap|ATSCtl|ATS|PRI|PASID|ACS|LnkSta:|[Aa]ccess denied|[Pp]ermission denied' \
  work/h2d-round7/pcie/*.txt > work/h2d-round7/pcie-compact.txt
```

仅保存读取结果，不运行 setpci 或修改驱动/内核设置。若没有 lspci、配置读取被拒绝、
出现 `<access denied>` 或某 BDF 没有输出，直接标为未取得，不能把缺失字段解释成 disabled。
请确认八个 BDF 都有输出再比较；命令失败也不需要为此启动或停止服务。

只需反馈四对路径的简短比较：

1. DevCtl 当前 **MaxPayload / MaxReadReq** 的字节数是否相同；DevCap 能力值另列。
2. ExtTag、10BitTag 的**支持状态与启用状态**是否相同，保留字段属于 DevCap/DevCtl 的上下文。
3. 能读取到的 ATSCtl、PRI/PASID、ACS、LnkSta 差异，以及所有 unavailable/access-denied 项。

优先贴出不一致的原始行；完全相同的部分可汇总为“相同”，无需上传全部配置。
配置差异本身不是因果证明，MRRS/Tag 能力也不等于实际 SM 事务大小或实测在途深度。
若可见配置一致，下一轮再隔离 host SM 读请求延迟和并行度；目前不增加 GPU 重跑。
