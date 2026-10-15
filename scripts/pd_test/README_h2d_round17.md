# 第十七轮：同载荷连续SM copy，CTA4/CTA8

分支仍为基于hash_verify的`perf/hisparse-h2d-diagnosis`。
本轮检验：**同为SM发起host读取，去掉稀疏gather组织后，能否接近同载荷CE？CTA4/8是否有不同响应？**
只用现有venv、CUDA JIT和pynvml，不需要NCU、Docker、新驱动或服务启动。

## 1. 五组及其用途

GPU4/5在同一个进程中顺序读取同一份NUMA1、Portable flags=1的512MiB host pool。
注册context仍为CUDA1。每次copy的前四组均为**1,868,800 B**，64个bank轮换。

| 方法 | 每copy有效字节 | 布局/组织 | 用途 |
|---|---:|---|---|
| gather4 | 1,868,800 | 原legacy CTA4/1024，B32/miss100，随机C4源/分页目标 | 当次基线，检查约31/23 GB/s差距是否仍在 |
| linear4 | 1,868,800 | 连续源→连续目标，CTA4/1024 | 连续SM参照 |
| linear8 | 1,868,800 | 与linear4相同，CTA8/1024 | 只检验连续kernel的grid影响 |
| ce_same | 1,868,800 | 与linear4/8完全相同的源views和目标 | 同尺寸CE参照，分开尺寸与引擎差别 |
| ce_bulk | 67,108,864 | 64MiB连续CE，7个bank | 保留约55 GB/s参照，校验NVML |

linear4/8及ce_same共享源地址、目标地址和目标guard allocation，源/目标均128B对齐。
连续源的64-bank工作集119,603,200 B；gather的逻辑字节量相同，但随机源地址覆盖更大范围。
这不是把同一组KV做生产packing：**连续组同时移除了plan/count遍历、C4 value/scale分离、
稀疏目标写入及其地址计算，改变了访问footprint。不能把全部收益归因于源连续。**

## 2. 诊断kernel及严格保持的部分

独立`hisparse_linear_probe.cuh`，不修改服务kernel、默认策略或原legacy JIT输入。
每warp遍历连续的96-word组，三条标量64位读取暂存后写回，尾部按实际word数保护。
读取复用生产`load_item_word<0>`，没有刻意引入128位向量或新的cache policy。
这是源程序组织描述，不冒充对实际SASS/在途事务数的测量。

CTA4/8使用**同一个JIT entry和同一份.so**，grid是运行时参数。
每卡gather4、linear4、linear8和ce_same还使用相同的bank序列重复次数：
不是给CTA8一个更大、更容易摊销开销的chunk graph。
记录的grid来自launch参数，未用profiler推断物理SM落点；CTA数不等于活跃SM数。

## 3. 测量与检查

复用第十六轮验证过的连续graph采集：每段只replay一次，graph内部无逐copy计时Event。
外围CUDA Event和墙钟照录；每阶段先跑相同copy负载2秒，测量2.5秒，裁掉首尾0.3秒。
CPU分配、正确性检查、编译及所有reset都在测量之外；不跑matmul预热。

正式顺序为device ABBA `[0,1,1,0]`。每卡两个波次：

- 第一波关闭NVML采样，方法顺序gather4→linear4→linear8→ce_same→ce_bulk。
- 第二波仅开启50ms Rx采样，方法顺序反转；状态只在窗口外读取。

总计20个窗口、约50秒正式测量，另有40秒copy预热、背景检查和setup开销。
OFF/ON与方法反序同时变化；这不是再次严格隔离采样效应的实验。
它用于暴露大幅时序/采样差异，CTA及方法比值分别在OFF和ON内计算，要求方向一致。

质量条件沿用Rx自检：提交/GPU区间≤0.10、GPU/wall为0.90–1.02、段耗时CV≤0.10、
后段/前段耗时0.90–1.10；ON至少20个样本、覆盖≥95%、Rx CV≤0.15、背景≤5%。
同方法ON/OFF归一化耗时必须0.90–1.10；bulk CE的名义Rx/有效吞吐0.85–1.15、有效吞吐≥45 GB/s。
**45 GB/s门槛只用于bulk CE，不用它否决同尺寸CE的真实固定开销。**

新kernel先检查8、264、584、1,868,800 B，CTA4/8，host/HBM源，两卡共32项：
每项eager和reset后的graph replay都逐字节检查payload及两侧guard。
边界测试刻意包含8B对齐但不16B对齐的地址；正式性能组则为128B对齐。
随后五组各自capture最终chunk graph，再reset/replay/同步/检查，10/10。

同源/目标/grid/chunk约束不满足或CUDA/字节检查异常立即停止。
main的窗口质量失败仍保留余下平衡顺序数据，完整实验仍标为`results_valid=False`。
报告按比较独立检查：`SM_valid`要求共享正确性/约束/JIT、完整20窗平衡顺序、
两卡gather4/linear4/linear8全部窗口有效且各自ON/OFF在门内。
满足时可输出`RATIOS_SM`和`CROSS_SM`；失效的ce_same窗口不参与这些SM比较。
`CEsame_valid`还要求ce_same有效；`bulk_reference_valid`还要求bulk窗口和NVML校验有效。
所有原始窗口质量标记及≤0.10提交门均保留，不给ce_same豁免。

## 4. 实机只做check，然后main

正常属主账号、仓库根目录同步：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

在现有gpuq分配物理GPU4/5的作业内：

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_linear.sh check "$PWD/work/h2d-round17/linear-check"
```

check只编译/检查，不采集正式性能。通过行应为：

```text
success=True mode=check edges=32/32 edge_valid=True graphs=10/10 graph_valid=True bindings=True JIT=True results_valid=True
```

身份physical4/5、host页在NUMA1也需核对。通过后直接继续：

```bash
bash scripts/pd_test/run_h2d_linear.sh main "$PWD/work/h2d-round17/linear-main"
```

main自动重复正确性和约束检查，避免工作区/JIT变化后沿用旧check。
完整通过期待相同正确性通过行，以及`TIMING valid=True windows=20/20 CE_calibration=True`。
若只有参考方法窗口失效，可检查`COMPARISONS`行使用独立通过的比较；不要将完整实验改判为通过。
任何阶段失败，保留输出，不安装工具、不调整门槛继续刷结果。
本地只验证了CPU逻辑/入口；没有GPU或nvcc，CUDA编译和执行由上述check验证。

## 5. 反馈与判读

只反馈check通过状态一行，以及main的**linear-compact.txt全文**（约25行）。
失败则给对应linear-compact.txt；文件未生成时给run.log最后30行。linear.json留在实机。
紧凑报告包括五组OFF/ON吞吐、Rx、质量指标、CTA8/4耗时比、linear/gather及linear/同尺寸CE比、卡间比。

- 连续SM接近同尺寸CE、卡间差距也明显缩小：说明原gather组织是重要限制，不能称SM host读取普遍被卡死在23 GB/s。
- 连续SM的CTA8明显快于CTA4、原gather仍不快：grid响应依赖工作组织，需解释gather的额外开销/访存形状为何抵消收益。
- 连续SM两档仍明显低于CE：本次连续实现没有达到CE参照；不能仅凭一个实现宣布SM硬件的绝对上限。
- gather基线差距不复现、OFF/ON不一致或质量不合格：先解释当次条件，停止作机制归因。

先看≥10%且两波同向的变化；小变化不立即扩大扫描。此轮没有生产packing、真实decode重叠或吞吐收益结论。

## 6. 首次正式结果与离线重解析（无需GPU）

实机check全部通过，main记录完整20窗；仅ce_same的四窗提交占比0.20–0.40越界，
因此完整`results_valid=False`保留。该占比是replay API耗时/GPU区间，不能当作GPU空闲占比。
其他四种方法在两卡OFF/ON全部窗口有效。旧报告将一个参考方法失败扩大到所有比较，
新版只修正报告范围，不重算或放宽窗口质量门。

以下取无采样OFF列，ON列检验方向：

| GPU | gather4 GB/s | linear4 GB/s | linear8 GB/s | linear4/gather耗时比 | CTA8/4耗时比 |
|---|---:|---:|---:|---:|---:|
| 4 | 30.8725 | 45.8514 | 45.8341 | 0.6733 | 1.0004 |
| 5 | 22.8927 | 33.1339 | 33.2250 | 0.6909 | 0.9973 |

ON对应linear4/gather耗时比0.6716/0.6997、CTA8/4耗时比1.0063/0.9941。
GPU5的ON相对OFF存在约2–3.4%耗时变化，仍在门内；不要写成完全无漂移。

- 当前连续SM实现的吞吐高于gather约43–49%，说明23/31 GB/s不是这些卡上SM host读取的普遍上限。
  但该实验同时改变plan遍历、源/目标布局、C4分离和footprint，不能将全部约30–33%耗时下降算作源连续的因果效应。
- 连续路径CTA4→8变化不足1%，当前实现单纯增加grid也没有明显收益。
  与gather的结果一致，只否定这两个具体实现下的4→8扩容，不能证明所有并行组织无效。
- 连续SM卡间GPU5/GPU4耗时比为OFF 1.3838、ON 1.4335；gather为1.3486/1.3759。
  卡间差距并非稀疏gather独有；不能再只用稀疏地址解释两卡差距。
- bulk CE约55 GB/s参照仍有效，但不是同尺寸对照。ce_same失效，暂不判断连续SM距同尺寸CE的差值。
  本轮没有证明硬件绝对上限、PCIe事务形状或某个硬件队列瓶颈。

同步分支后，只在原venv中对已有JSON运行；无需gpuq、CUDA_VISIBLE_DEVICES、numactl或GPU重测：

```bash
PYTHONPATH=python "$LAGOM_PY" scripts/pd_test/reanalyze_h2d_linear.py \
  --report work/h2d-round17/linear-main/linear.json \
  --output-dir work/h2d-round17/linear-reanalyzed
```

使用新目录。输出保留输入和分析器SHA256，不改原JSON；退出0仅表示SM比较有效，
不表示完整实验通过。预期`results_valid=False`且
`COMPARISONS SM_valid=True CEsame_valid=False bulk_reference_valid=True`。
只需反馈该行、两条`RATIOS_SM`及一条`CROSS_SM`，不需要原始日志或额外GPU实验。
