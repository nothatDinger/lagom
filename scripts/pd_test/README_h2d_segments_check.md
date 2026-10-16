# Copy区段细分：仅第一阶段正确性与SASS

目标：给CTA4→8每KV copy区段耗时增长建立可核对的计时边界。
本提交无性能入口、不产生带宽结论、不改生产CUDA kernel或服务dispatch。
仅物理GPU4；不跑NCU、不改频率，不继续CE压力测试。

## 三个产物

- legacy：原生产kernel，同一JIT SHA用于对照。
- control：复刻原request/count遍历、原copy_miss_item，无时间戳和记录写出。
- trace：同一任务分配，将copy区段拆成T0/T1/T2/T3/T4。

T0在读取src/dst plan之前；T1在C4源/目的地址准备后；三条原legacy标量load后是T2；
三组结果经XOR消费后是T3；原lane分工的三组store后是T4。
T1以六个有效地址的OR作为谓词输入，约束地址准备；T3前XOR校验值累计写到记录，
使结果消费保持live。两者仍须SASS核对，不能凭PTX文本认定调度顺序成立。
不加系统fence、warp/CTA同步、软件队列、128bit向量化或cache policy变化。
T4不是store排空完成时间。

记录每warp一行15个int64：cta/local_warp/global_warp/smid/begin/end/
copy_sum/kv/visits/prep/issue/residual/store/checksum/bad_order。
每段先逐KV求差再累计；copy_sum=四段之和；只lane0写出该warp记录。
检查实际KV数量、访问数、guard、非负段时间、总时间范围及时间顺序。
零长度区段允许timer分辨率影响，不把零解释成无等待。
checksum仅用于维持数据依赖，不代替输出逐字节校验。

## 第一阶段检查

legacy/control/trace × CTA4/8 × host/HBM =12项。
复用原边界fixture：非连续plan行、页边界、多个request、0及多miss、padding；
eager及同一graph内更改counts/real/source/plan后6次重放，逐字节与未触碰区域检查。
全程只验证功能和记录自洽，不报告性能。HBM只作正确性覆盖。

已加载产物SHA写到check.json；现有cuobjdump离线导出三份完整SASS及资源信息，
记录timer位置及完整控制字解码。success只表示提取完整；
performance_ready=False、timing_semantics_reviewed=False是预期值。

## 必须人工核对

1. trace的T0前没有提前读取本项plan；T1前地址依赖已完成，host load均在其后。
2. T1→T2确有三组host LDG.E.64，lane布局与legacy一致。不能把GPU plan的LDG计入。
3. 解码所有沿途控制字、分支路径，核对T2之前是否已等待host结果。
   若提前等待，放弃T2细分，只评估T1→T3粗区段，不能硬称发射/返回分离。
4. 三组结果的消费XOR确实在T3前，相关scoreboard等待覆盖三组结果；
   若消费被移动到T3后，当前候选不具备读取完成计时资格，需要修改。
5. 三组STG在T3→T4间；无提前/延后跨界，无fence；T4不证明写入排空。
6. lane0的记录含正确依赖路径；检查谓词分支和所有辅助回跳，不能只贴几条LDG。
7. control与legacy的访存形状一致，trace无stack/local spill；报告寄存器数。
   如spill或边界失败，先修探针或退回粗粒度计时，不跑性能。

即使通过，本探针仍可能改变发射间隔与寄存器压力；下一阶段必须配对验证
control/legacy与trace扰动，才能解释CTA4/8分段差异。本阶段不设置硬件队列结论。

## 运行与反馈

属主同步分支，在gpuq分配的GPU4内：

```bash
cd /home/jovyan/td69032/lagom_h2d_latency_test
LAGOM_PY="$PWD/.venv/bin/python3" bash scripts/pd_test/run_h2d_segments_check.sh \
 "$PWD/work/h2d-segments/check-v1"
```

脚本只做check和离线提取，结束即停，没有main参数。
反馈：check/check-compact.txt、codegen/codegen-compact.txt、
codegen/trace.sass.txt和codegen/control.sass.txt（完整保留机器码/控制字）。
legacy.sass.txt与check.json/codegen.json留本地，需要时再补。
失败回已有compact及相应run.log或codegen.log最后30行，不继续GPU作业。

本地无GPU/nvcc，CPU校验与语法通过不能代替实机CUDA编译、正确性或SASS审查。

## 第一版实机审查：暂不具备原kernel性能解释资格

check 12/12（全部eager、graph6/6、padding/guard）通过；legacy/control/trace寄存器
34/34/53，STACK/LOCAL/SHARED均0。三个JIT SHA：
- legacy 1bb816c524eb665041b323c5826dcc849ea15495d3a45aa207c9cf40154ffa6c
- control d2ee0815c3de057c2ef3289e913bce9827fa95fa344a6178b34f96bb28a10fff
- trace 29fa663f9617f602567ca5100bc9239f0a08c59bca85997a5db446377f52fa6c

第一版trace的关键边界（仅适用于此SHA）：
- T0=0590；plan LDG=0630/0650，在T0后。
- T1=0ab0；host LDG=0ac0/0ad0/0ae0，全为W5、M0；前后至T2无M0x20等待。
- T2=0b10；0b20 LOP3 XOR消费三个load低半结果，M0x20等待barrier5；
  0b30消费三个高半结果；T3=0b40位于其后，M0x02是读取地址寄存器的barrier1。
- STG=0b50/0b80/0b90；T4=0ba0，M0x02可能包含store源寄存器读取依赖等待，
  不是写入HBM排空证明。期间还包含checksum累计指令，不是纯store机器指令耗时。
- T2分隔的是可见发射区段和残余依赖等待，不代表请求已经进入PCIe或精确队列入队。

阻断项：control仍为三条LDG.E.64.CONSTANT，而trace三条变成LDG.E.64；
即使边界有效，访存语义已变化，不能解释原legacy的CTA4/8增长。
trace中load顺序也变为tail、prefix0、prefix1，说明源码顺序不能当机器码证据。

修复候选仅把trace三个C++读取换成显式__ldg，要求新SASS恢复.CONSTANT；
不改生产kernel、不改cache到.ca、不追加性能。__ldg只是候选，必须重新核对
三条load修饰符、完整wait mask、T0..T4顺序和资源。新codegen compact增加MEMORY行。
用check-v2新目录执行同一检查脚本，回全部四份文件；旧check-v1保留。
