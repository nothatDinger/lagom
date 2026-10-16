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
