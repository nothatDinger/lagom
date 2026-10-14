# 第十三轮：跨请求 D1/D2 候选的正确性与编译产物检查

沿用 `perf/hisparse-h2d-diagnosis`，基于 `hash_verify`。
本轮只编译、验证字节正确性并离线提取 SASS；没有性能计时、模型服务或 NCU。
使用已有 CUDA/venv/cuobjdump，无需安装新工具或 Docker。

修正版实机正确性与SASS已核对，后续操作见[第十四轮配对性能对照](README_h2d_round14.md)。
补充文件中的0950 BRA有M0x02，但未等待A写入的barrier 5；不能描述为“中间完全无非零wait”。

## 1. 为什么现在做这个候选

[第十二轮分析](README_h2d_prefetch_analysis.md)已确认：相邻 KV 的源数据没有算法依赖，
但 legacy 的同 warp 下一 KV load 排在当前 KV 的依赖 store 后。
B32/miss100 下每 warp 有25条 KV、相邻任务全部跨请求，简单展开 request 内循环无效。
这给出一个具体可干预的代码限制；它损失了多少带宽尚未测得。

| 组 | 遍历 | 同 warp 的源码读取/写出组织 | 后续比较用途 |
|---|---|---|---|
| legacy | 原实现 | 原实现 | 已知基线 |
| iterator-D1 | 跨 request 游标 | load(A) → store(A) | 识别遍历重写/编译变化 |
| iterator-D2 | 与 D1 共用游标 | load(A) → load(B) → store(A) → store(B) | 相对 D1 识别两项暂存的增量 |

当前修正版在上述 value load 前先完成本组一条/两条 KV 的游标推进和 GPU plan 准备。
D2 的两项有效分支在 host load 前选择；该分支的两次 host 读取之间没有源码级
`next()`、`SHFL` 或逐项有效性判断。单项尾部使用与D1相同的路径。
编译器仍可能重排，不能以此替代新 SASS 核对。

游标保留原 warp 的扁平任务序列 `w, w+128, ...`，counts 仍按32个请求的窗口读取和分发。
两项分组可跨 request/32-request 窗口，末尾不足两项时只读写有效项。
两候选使用相同 legacy C++ scalar load、lane分工、C4 value/scale地址和 CTA4/1024。
没有源打包、改 cache policy 或增大线程数。

新增独立 `hisparse_copy_iterator.cuh` 和 `copy_cache_planned_mla_iterator(depth=1|2)` 诊断入口；
服务调用路径仍用 legacy，没有环境变量切换。原 `hisparse.cuh`、`kvcacheio.cuh` 未修改。
这一步用独立 JIT 编译产物检查假设，不把源码写成“两项”就当作硬件已做到重叠。

### 首版反馈与修正依据（0c50eaeec）

首版实机正确性12/12，每项graph 6/6全部通过；legacy/D1/D2分别34/26/32寄存器，
LOCAL、STACK均为0。D2有六条source LDG排在首个STG之前，但这个条件不足以通过。
用户补充的完整高位控制字段给出以下明确对应（位编号相对指令的高64-bit word）：

| 位置 | 高64-bit word或其高24位 | write barrier | read barrier | wait mask |
|---|---|---:|---:|---|
| A：0530、0540 | `0x000368000c1e9b00` | 5 | 1 | 0 |
| A：0550 | `0x000f62000c1e9b00` | 5 | 7（无） | 0 |
| 0770 SHFL | 高24位 `0222a4` | 2 | 1 | `0x22`，即barrier 1、5 |
| B：0a20、0a30、0a40 | 随后发起source load | — | — | — |

字段为 `W=(word>>46)&7`、`R=(word>>49)&7`、`M=(word>>52)&0x3f`。
0770虽只读GPU count的R6，附带的wait mask仍包含A source load所用的barrier 5。
在经过0770的路径上，B发起前要求对应依赖就绪；如果更早已经等完，也不能恢复A/B读取重叠。
这不意味着0770一定产生可测停顿：A可能已经自然完成。此处证明的是强制完成约束。

B32/miss100的相邻任务全部跨request，需要再次执行counts分发SHFL，因而这个路径对目标载荷重要。
同request多miss的路径可能跳过它，不能把首版所有执行路径都判为串行。
**首版不进入本轮性能对照**：其结果无法干净检验目标跨request读取重叠。
这是新增候选的调度问题，不能当作原服务23 GB/s的根因或预取无效的证据。
修正版把metadata准备移到host load之前，先重新验证正确性和实际codegen；仍使用既有工具。

## 2. 正确性覆盖

三组 × host/HBM 源 × 两卡，共 **12项**；每项含 eager 检查和同一 CUDA graph 的6种输入状态。
host 使用 PyTorch pinned allocation，每卡分别创建；本轮只验证执行，不比较带宽。
每个设备显式创建本设备 capture stream，报告对应设备，避免之前的跨卡默认 stream 问题。

fixture 有65个有效请求及第66个 padding 请求，plan 行宽1272、实际 stride `(1280,1)`。
不同请求包含0、1、65、100、127、128、129及1272条 miss，强制覆盖 C4 的63/64槽边界。
padding 请求的 count 故意非零、plan ID 为 -1，必须受 `num_real_reqs` 屏蔽。

6种 replay 状态：real=0、全零miss、B32/miss100、跨32请求边界的real=33、
跨64请求边界的real=65混合载荷、real=1且更换源plan。
每次 replay 前改变 host 字节，HBM 对照同步刷新，排除旧结果或旧plan恰好通过。
独立 CPU oracle 检查整个目的 buffer，包含576B value、8B scale、页padding及全部未写区域。
校验失败即停止并记录 active case；没有继续测速的步骤。

## 3. 实机只执行这一条检查作业

在正常属主账号中同步分支，保留已有本地修改，不清 JIT cache：

```bash
git fetch origin perf/hisparse-h2d-diagnosis
git merge --ff-only FETCH_HEAD
export LAGOM_PY=/home/jovyan/td69032/lagom_h2d_latency_test/.venv/bin/python3
```

通过现有 gpuq 分配物理 GPU4/5，在作业内执行以下命令。
脚本设置可见设备、绑定NUMA1、保留两个阶段的日志。
若 `cuobjdump` 不在 PATH，可设置 `H2D_CUOBJDUMP=/已有/CUDA/bin/cuobjdump`；不安装新版本。

```bash
bash scripts/pd_test/run_h2d_iterator_check.sh "$PWD/work/h2d-round13/iterator-check-v2"
```

目录必须全新；重复执行时换目录名。首次运行包含 JIT 编译时间。
wrapper 先执行 `check_h2d_iterator.py`，成功后才执行离线 `collect_h2d_iterator_codegen.py`。
提取器校验进程实际加载的三个 `.so` 的 SHA256，使用 `--dump-sass` 和 `--dump-resource-usage`，
不传不兼容的 `--demangle`。git 的 dubious ownership 仅记录为元数据错误，不中断已完成的结果。

正常输出应包含：

- `success=True correctness=12/12 artifacts=3/3`。
- 12条 CHECK 均 `eager=True graph=6/6 all_bytes=True pinned=True stride=[1280,1]`；
  `capture_device` 与逻辑设备一致，共72次 graph 结果检查。
- 身份为物理GPU4（98:00.0）/GPU5（b8:00.0）。
- `CODEGEN success=True artifacts=3/3 performance_ready=False review_required=True`。
- 各组 `CTRL_LOAD64 ... complete=True`：完整高64-bit控制字全部解析；
  `CTRL_WAITS` 给出全kernel非零wait mask及DEPBAR，包含地址区间外的跳转辅助块。

**`performance_ready=False` 是有意保留的人工判读状态，不是失败。**
CODEGEN success 只说明三份产物已提取，不宣称 D2 生成了目标指令顺序。

## 4. 本轮回传与 SASS 判读

请只回传两个短文件：

- `check/iterator-compact.txt`：正确性、capture设备、身份。
- `codegen/codegen-compact.txt`：寄存器/local/stack、load/store形状及关键指令顺序。
  现在同时给出 `CTRL_LOAD64`（W/R/M及完整高64-bit word）与 `CTRL_WAITS`；原样回传即可，
  不需要手工截断机器码或自行给出“无等待”的结论。

其余 `.so`、JSON、SASS 留在本地。失败时给对应 `check.log` 或 `codegen.log` 尾部约30行及
active case，无需重跑或上传整份日志。若摘要不足以确认依赖，再从现成文本提取指定片段即可。

性能运行前需要核对：

1. D1/D2 确实使用同样的 source load 形状/策略，D2 的 A/B 是不同有效 KV；
   不能把读取 GPU plan 当成第二条 host KV。
2. **第二条 KV 的 source load 提前到第一条 KV 的等待/消费之前。**
   不只看 STG：若中间 MOV、计算、控制码等待或分支重汇合先要求 A 结果就绪，
   则“六条 LDG 都排在第一个 STG 前”仍不足以证明目标读取重叠。
3. 分支/谓词覆盖两项有效路径和单项尾部；source load 的寄存器流向应能对应各自目的 store。
4. 记录寄存器数；要求资源信息完整，未出现意外 `LOCAL/STACK`、`LDL/STL`。
   若有 spill 或明显额外串行化，先审查/修正候选，不进入性能比较。

`.order.txt` 保留 memory/branch 指令及机器码；`.read-window.txt` 另外保留首个64-bit load
前到最后store后的**连续指令**，便于检查被摘要省略的消费者与等待。
`read64_before_first_store_static_hint` 只是定位提示，可能包含plan读取或互斥路径，
不是自动通过条件，也不是动态事务数或硬件队列深度。
控制字段目前只解码SM90/SM90a；格式缺失/截断或其他架构保留unknown，不推断无等待。
`CTRL_WAITS` 也不自动判为失败：metadata准备阶段的等待及store前的必要等待都合理，
必须结合分支路径、barrier分配/复用和value寄存器判断。即便排除了强制等待，
也只能说明允许跨KV重叠；是否实际重叠及收益仍需要后续测量。

本轮执行到提取完毕即可。确认正确性及 codegen 后，再准备一次固定分散源、
B32/miss100主目标、host/HBM对照的三组配对性能实验；不预先安排线程/深度扫描。
D2/D1回答增加暂存是否有用，D2/legacy回答净收益；有收益也仍需最终真实decode验证。

## 5. 开发环境验证边界

CPU 遍历核对的4,204组现在同时比较 legacy、D1、D2 与独立扁平oracle。
CPU 单测覆盖字节oracle、尾项/padding、API隔离、实际JIT产物识别、SASS提取和资源信息缺失。
修正版26项CPU测试通过，包含两项新增控制字段测试，覆盖已提供的legacy/D2机器码及缺失字段。
Python静态检查、两个CLI加载及shell语法检查通过；
6项CUDA测试因无GPU跳过，没有计入通过数。
开发环境没有 NVCC/GPU，不能验证 CUDA 编译、真实寄存器分配或 graph 执行；
该段记录开发时的验证边界；随后修正版的实机12/12与新SASS核对结果已归档到
[第十四轮§1](README_h2d_round14.md#1-第十三轮判读结果)，不以CPU通过代替CUDA验证。
