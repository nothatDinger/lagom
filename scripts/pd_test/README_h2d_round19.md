# 第十九轮：相邻 C4 的直接索引对照

分支 `perf/hisparse-h2d-diagnosis`，基于 hash_verify。
问题：第18轮相邻 gather 到 linear 仍有约18–20%的耗时差，去掉 plan/count 组织后能恢复多少？

## 1. 三组，仍是12个窗口

| 方法 | 地址/处理 |
|---|---|
| gather_adjacent4 | 第18轮相邻 C4 gather，原 legacy kernel，保留请求遍历与 plan/count 读取 |
| direct4 | **同一份源与目标 allocation、相同的逐KV地址和warp访问顺序**；直接计算索引后调用原 `transfer_item<0>` |
| linear4 | 第17/18轮连续字节基线；同内容、同字节量，保留作参照 |

固定 B32/miss100、CTA4/1024、64 banks、1,868,800 B/copy。复用第18轮CPU packing和独立原始源oracle。
不重测随机 gather，不测 CE、Rx，不加 CTA，不跑 NCU，不改服务 kernel 或默认配置。
每卡先正序、再反序；设备顺序 `[0,1,1,0]`，每窗口copy预热2秒、测量2.5秒，裁去首尾0.3秒。
事件为空的内部graph、单次提交、相同chunk工作量及质量门沿用第18轮。

直接索引的新内核只用于诊断，源码为 `hisparse_c4_direct_probe.cuh`：

```text
warp = blockIdx.x*32 + threadIdx.x/32
for flat = warp; flat < 3200; flat += 128:
    transfer_item<0>(output, source, dst_index=flat, src_index=bank_start+flat)
```

原相邻计划中 `(request,miss)` 的 flat ordinal 为 `request*100+miss`。
CPU 对128个warp逐项比较原请求循环和上式的**完整顺序**，每warp均25项。
保留每warp每轮一条584B KV、C4的576B value/8B scale分离和原lane分工；不增加预取stage。
目标指针直接借用相邻gather的buffer，校验仍对照**原始随机源**，不以准备后的数据自证正确。

## 2. 先检查新内核和编译产物

check在物理GPU4/5上分别验证 `count=0/1/65/3200`、host/HBM源，共16项：
src_start=63、dst_start=9，覆盖跨C4页和不足一轮128warp的尾部。
每项 eager 和 reset 后 graph replay 都检查全部输出、未触碰字节和原源数据。
所有capture均显式使用本卡stream；随后三组最终chunk graph另做6项reset/replay校验。

check脚本自动用**现有CUDA的cuobjdump**提取实际加载的 legacy/direct 两个库，不发GPU作业：

- 库路径和SHA256来自本次进程的映射，提取前后核对hash。
- 两者均须有 `LDG.E.64.CONSTANT ×3`、`STG.E.64 ×3`，无128位向量、LOCAL/STACK spill。
- 记录REG及完整SASS/resource文本；不传 `--demangle`，不要求新版NCU。
- 该门只核对静态访存形状/资源；不声称机器码全部相同或测出了PCIe请求数。

若编译产物不同，保留dump并停下，先回传短报告，不进入main或自行放宽门。
main必须提供该check产生的codegen.json；实际加载legacy/direct的hash须匹配，才允许正式窗口开始。
REG/控制流仍可能随移除plan而变化；同grid不保证同物理SM落点。本轮测的是组织改写的净效果。

## 3. 执行命令

正常属主账号在仓库根目录同步：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
```

在现有gpuq分配物理GPU4/5的作业内：

```bash
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
bash scripts/pd_test/run_h2d_direct.sh check "$PWD/work/h2d-round19/direct-check"
```

期待：

```text
success=True mode=check correctness=6/6 ... graphs=6/6 ... packing=True bindings=True JIT=True results_valid=True
DIRECT_EDGE valid=True checked=16/16 schedule={'valid': True, 'warps': 128, 'items': 3200, 'ordered_items_match': True}
CODEGEN success=True artifacts=2/2 scalar_shape_valid=True
```

同时核对physical4/5和host NUMA1。以上全部通过后直接运行：

```bash
bash scripts/pd_test/run_h2d_direct.sh main "$PWD/work/h2d-round19/direct-main" \
  "$PWD/work/h2d-round19/direct-check/codegen/codegen.json"
```

若PATH找不到已安装的cuobjdump，可在check前设置 `H2D_CUOBJDUMP=/现有CUDA/bin/cuobjdump`，不需要安装工具。
正式组要求 `CODEGEN_EVIDENCE valid=True`、`TIMING valid=True windows=12/12`、`results_valid=True`。
本地无GPU/nvcc：CPU和入口检查不代替实机编译、字节检查或SASS门。

## 4. 反馈及判断

- check只反馈通过行、DIRECT_EDGE行及codegen-compact.txt（约6行），确认身份/NUMA。
- main反馈bridge-compact.txt全文，约22行；原JSON和SASS留在实机。
- 失败停下，反馈对应compact；未生成则提供run.log或codegen-run.log尾30行。

重点是两卡两轮的 `direct/adjacent_us`；另给 `linear/direct_us` 和 `linear/adjacent_us`。
约≥10%且两轮同向才继续推进。
明显加速支持相邻布局下plan相关组织形成限制；无收益降低该方向优先级。
差值不能当成可逐项相加的“plan独占耗时”，也不承诺原分散源/真实服务有同幅度收益。
剩余差额再用于选择每KV处理方向；本轮不新增并发或cache策略扫描。
