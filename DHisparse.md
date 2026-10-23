# DHisparse 实现与验证记录

论文：[Adaptive DHiSparse，arXiv:2610.13708](https://arxiv.org/pdf/2610.13708)。
分支 `DHisparse` 基于 `swapin_oneshot` 的
`8948c57fb566fd99fae2ae01f79c6c3f15b7a408`。

**状态：部分复现，尚未完成整篇论文的端到端复现。** 以下“已接入”表示代码路径存在，
并不表示已经在论文的模型和硬件上验证。当前执行环境只有 CPU PyTorch，没有 NVIDIA GPU。

## 论文功能对应

| 论文功能 | 当前状态 | 代码入口 |
| --- | --- | --- |
| 必需 hot/tail/staging 与可选历史分离 | 已接入；可选区仅保存导入历史的完整页 | `srt/mem_cache/dhisparse.py`、`srt/managers/dhisparse_coordinator.py` |
| 本地 prefill 保留原 GPU 历史、延迟主机备份 | 已接入；资源不足时回退先备份再释放的路径 | `AdaptiveHiSparse.admit_resident` |
| 保留 admission reserve、按请求回收、有限轮转回填 | 已接入；回收等待备份及读者，完成拷贝才发布 ready map | `ResidencyManager` |
| tail / ready history / bounded hot / host miss 解析 | 已接入；全部直接命中时 CUDA 内核跳过 hot 哈希构造 | `kernels/jit/csrc/dhisparse.cuh` |
| 多 query union 去重、查询顺序和因果边界 | 已接入；同一 request/source 的重复 host miss 只搬运一次 | `kernels/ops/kvcache/dhisparse.py` |
| 验证溢出 staging 与跨层组复用 | 已接入；容量按最大验证宽度计算，attention 后复制提升到固定 hot 地址 | `VerificationLayout`、`allocate_payload_slab` |
| GLM IndexShare 的精确后续层预取 | 已接入；共享索引的层使用各自 payload 和并存 arena，事件连接预取与读者 | `AdaptiveHiSparse.resolve` |
| DFLASH/DSPARK 接受前缀提交与异常回滚 | 已接入 DSA 事务；保留并扩展原分支 C4 压缩边界事务 | `prepare_verify` / `commit_verify`、`prepare_dspark_verify_window` |
| 字节一致的量化 KV 搬运 | 已实现线性布局和 DSv4 的 576-byte value + 8-byte scale 布局 | `dhisparse.cuh`、CUDA payload 测试 |
| DFLASH 物理草稿环形缓存 | 已接入窗口化 DFLASH；`R = page_round(W + p - 1 + Kmax)`，例如 2176 行 | `DraftRingLayout`、`dflash_worker_v2.py` |
| DFlash2 提议分布 | 已加入候选 transition selector、分组卷积、稀疏 q 到概率验证的接入；待模型验证 | `srt/models/dflash2.py`、`dflash_worker_v2.py` |
| DSv4.1 Full / Reindex / Reuse 物理源共享 | **未完成**：基线缺少 V4.1 模型、索引器和存储布局，不能用 V4 名称替代 | 待移植模型栈和物理源生命周期 |
| 借用空闲 index backing 存储可选历史 | **未完成**：当前可选历史只使用专用 payload 页 | 待实现借出、索引分配时收回与地址解析 |
| PD 接收端 resident-first 导入 | **未完成**：PD 仍直接写 host；接收后可回填 GPU | 待扩展传输地址和接收端 admission 协议 |
| DSpark 专用物理草稿 ring | **未完成**：其现有 SWA/unified 路径保留；新的 DFLASH ring 不冒充此功能 | 待接入 DSpark pool、attention metadata、hidden injection |
| 论文端到端正确性、显存与吞吐实验 | **未完成** | 需要论文模型权重、GPU 和 PD 部署 |

DFlash2 数学实现参考 SGLang 提交
`6e667c431009dfd70d88ea2dd9f8784bd0787d89` 的候选选择器与分组卷积。
这里只移植相关语义；没有移植其 fused selector 性能路径，也没有合并整个上游模型栈。

## 配置与行为变化

启用 `--enable-hisparse` 时，`adaptive` 默认为 `true`。

```json
{
  "adaptive": true,
  "top_k": 2048,
  "device_buffer_size": 4096,
  "host_to_device_ratio": 2,
  "resident_first": true,
  "admission_reserve_pages": 1,
  "refill_pages_per_step": 16
}
```

- hot 容量按单 query 配置，验证 union 不再要求全部放入 hot。
- staging、元数据和 DFLASH ring 从 KV 预算中预留；超预算在初始化时明确报错。
- `force_h2d_each_step`、命名预测预取器和跳过 KV IO 的调试模式与 adaptive 冲突，会报错。
- `adaptive: false` 保留原分支实验路径；原分支 speculative 的其他约束仍然生效。
- 当前新 CUDA resolver 不支持 ROCm；DSA speculative 接入限于线性 DFLASH/DSPARK。
- compact/ragged 验证保留 eager 执行限制；带 grammar 的 speculative adaptive 请求拒绝执行，
  因为该路径尚未把 scheduler 最终截断长度接入 KV 提交。
- resident-first 回收使用同步等待确保读者结束，尚未做无阻塞回收优化。

## 验证

2026-10-23，本地执行：

- 139 个测试通过、4 个 subtest 通过，16 个因环境/设备条件跳过。
- 覆盖既有 HiSparse allocator、验证事务、EAGLE/DSpark、pool budget、预取回归，以及新增
  union/overflow、因果屏蔽、byte payload、ring wrap、arena alias、回收顺序、ready 发布、
  DFlash2 概率和卷积边界测试。
- CUDA 13.4 对线性与 packed resolve/promote 模板完成 `sm_90` 和 `sm_103a` 离线编译。
- 通过仓库的 isort、Black 和 Ruff `F401,F821,UP037` 检查。
- **没有执行 GPU 内核、CUDA Graph replay、真实模型推理或性能测试。**

新增 CPU 测试：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/unit/mem_cache/test_dhisparse.py \
  test/registered/unit/managers/test_dhisparse_transactions.py \
  test/registered/unit/spec/test_dhisparse_dflash2.py
```

在安装本仓库 CUDA 依赖的 NVIDIA 环境执行字节与 graph replay 检查：

```bash
PYTHONPATH=python python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_dhisparse_cuda.py
```

模型集成检查应先使用单机 DSA + DFlash2、`--disable-radix-cache`、
`--speculative-algorithm DFLASH --speculative-num-draft-tokens 8`
和 `--speculative-draft-window-size 2048`，显式设置可容纳预留内存的
`--max-running-requests`。先关闭 CUDA Graph 验证完整推理，再验证静态 graph 路径。
这里没有提供“已验证的论文启动命令”，因为模型和 GPU 测试尚未运行。

完整验收还需：移植 V4.1 源共享；完成 index backing 借用和 PD resident-first；接入 DSpark ring；
在相同模型、稀疏选择、提议/接受参数与负载下比较输出、接受长度、KV 字节、HBM 峰值、
host transfer、TTFT、TPOT 和吞吐。当前结果不能用于宣称论文加速比。
