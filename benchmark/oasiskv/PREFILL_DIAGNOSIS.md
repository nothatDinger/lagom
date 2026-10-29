# 区分 prefill 写入开销与 decode 并发性能

这份诊断对应 `30d373987` 的 GPU 报告：16K 输入、32 请求、256 输出，
OasisKV 的 `output_tps` 在 C=1/2/4/8 为 31.01/28.38/33.88/33.23；
它是原生 serving bench 的端到端输出吞吐，包含 prefill、排队和排空。
本次 append 批量化只做了 CPU/Triton 数值验证和 SM90 离线编译，尚无 GPU 加速结论。

## 已确认的开销和修改

原 `append()` 对每个 block 单独提交 host backup、QUEST min/max 归约、摘要赋值以及 GPU 页写入。
8192-token chunk / block_size=16 有 512 个 block，因此 Python 和小 kernel 数量随 chunk 的 block 数增长。
GPU 报告的每调用 53–62 ms、每请求约64次调用，对应数秒额外 TTFT，和源码一致。

CUDA append 现在采用：

- 一个 `append_chunk` Triton launch，按 head/block/维度分片，更新全部 QUEST 摘要和需要写入的 GPU 页；部分块与已有 min/max 合并，保留 active block 的清零语义。
- 每个 KV head 一次连续 pinned-host DMA。Qwen3-8B、TP=1、8个KV head时是8次当前 chunk backup，不再是512次按块 copy。
- CPU 保留旧实现作为参考；decode full graph 的 `append_token` 路径继续使用原来的 kernel。eager append 也使用批量写入。

物理页的首次分配/清零、dense→sparse 转换、host扩容、数据打包和PCIe字节量仍需测量。
prefill attention 继续直接使用 GPU 当前 chunk；这次没有重新引入当前 chunk 的 D2H→H2D 往返。

## 先跑写入 microbenchmark

从仓库根目录、在同一 GPU 和 NUMA 绑定下执行。脚本先比较完整 KV、摘要和页内容，再交错测量旧/新实现：

```bash
PYTHONPATH=python python benchmark/oasiskv/bench_append.py \
  --heads 8 --key-dim 128 --value-dim 128 --dtype bfloat16 \
  --context-length 16384 --chunk-size 8192 --top-k 2048 --block-size 16 \
  --scope write --output /srv/oasiskv/append-write-new.json

PYTHONPATH=python python benchmark/oasiskv/bench_append.py \
  --heads 8 --key-dim 128 --value-dim 128 --dtype bfloat16 \
  --context-length 16384 --chunk-size 8192 --top-k 2048 --block-size 16 \
  --scope append --output /srv/oasiskv/append-total-new.json
```

`write` 排除 allocation/growth/packing，接近报告中的 block-loop 范围；`append` 包含它们。
两种范围都等待真实 D2H 完成。每次测量都是一个 layer，不包含 attention、draft model 或32层完整服务。
`host_call_ms` 是 CPU 调用墙钟，可能含内部同步；`stream_elapsed_ms` 的 CUDA event 区间也可能含 CPU 提交不及时造成的空闲，不能叫纯 kernel 时间。
`completion_ms` 给出调用直到完成的墙钟时间。不要把任一项乘32当成已测得的 TTFT 改善。

## 吞吐平台目前不能归因为 decode 天花板

相同 PrefillAdder 不等于相同服务时间：同一种调度策略下，OasisKV 较长的 prefill 仍会延迟其它请求的 decode。
固定串行成本即使不随 C 增长，也能限制端到端吞吐；“每次 append 只增长1.16倍”不能排除它。
但若只有每请求3.5秒这项成本，256/3.5≈73 tok/s，尚不足以单独解释33 tok/s，仍有其他成本待分解。

`1000 / tpot_mean_ms` 不是服务器 decode TPS；它至多是每请求平均出 token 速率的近似，且“均值的倒数”也不等于速率的均值。
TPOT 区间包含首 token 后的调度暂停和其它请求的 prefill，所以 13.57→142.67 ms 是明确的用户时延退化，不能直接推导为纯 decode 吞吐下降10.5倍。
`running_batch_mean` 是按 forward 样本平均，不能直接乘到 TPOT 倒数上还原服务器吞吐。

C=1 的数据本身是自洽的：TTFT 4.79秒 + 255 × 13.57毫秒 ≈ 8.25秒/请求，
256 / 8.25 ≈ 31.0 tok/s，和测得的31.01一致。33 tok/s 是这个短输出负载的真实服务吞吐，
不是无效测量；只是不能把它称作纯 decode 的能力上限。

## 256 / 2048 输出长度对照

先在同一份代码上做这项实验，避免把输出变长和 kernel 更新两个变量混为一谈。
基线若有未提交的端口/清理修补，应保留并记录 diff；不要为了切换提交丢弃这些修补。
本次只改计算路径和增加独立脚本，没有覆盖执行 agent 的 `run.py` 改动。

为每档生成独立配置，以下复制默认配置中的模型、draft、数据、graph与KV参数。设备号按作业分配调整：

```bash
python - <<'PY'
import json
from pathlib import Path
base = json.loads(Path('benchmark/oasiskv/config.json').read_text())
for length in (256, 2048):
    config = json.loads(json.dumps(base))
    config.update(input_lengths=[16384], output_length=length,
                  num_requests=32, concurrencies=[1, 8], repeats=3)
    config['models']['qwen3-8b']['cuda_visible_devices'] = '0'
    path = Path(f'/tmp/oasiskv-length-{length}.json')
    with path.open('x') as stream:
        json.dump(config, stream, indent=2)
PY

env -u CUDA_LAUNCH_BLOCKING python benchmark/oasiskv/run.py run \
  --client sglang-bench --config /tmp/oasiskv-length-256.json --models qwen3-8b \
  --output /srv/oasiskv/length-256-new

env -u CUDA_LAUNCH_BLOCKING python benchmark/oasiskv/run.py run \
  --client sglang-bench --config /tmp/oasiskv-length-2048.json --models qwen3-8b \
  --output /srv/oasiskv/length-2048-new
```

这是小规模定位实验，不替代 Fig.11 协议的1024请求与完整并发扫描。
保留 `num_requests=32`、输入、seed、设备、代码、客户端及server配置，只改变输出长度；同时测 dense 和 OasisKV。
不要用 `--smoke`，它会改成2请求、C=1、输出32。

每点报告 output_tps、TTFT、TPOT、实际 running batch、duration和完成token数。
2048输出时TPS上升支持 prefill 被摊薄，但仍需比较 decode trace；TPS不升也不能仅凭一个点断言硬件上限。
在新代码上复测256配置，才能单独估计 append 修改对 TTFT 和端到端TPS的影响。

## 获取不含 prefill 插入的 decode 证据

在2048输出的C=8运行中，选取**没有 prefill、实际 running batch稳定、无尾部排空**的连续时间段，
对齐 SSE/实际输出 token 计数与 GPU 时间线，报告区间输出 token / 墙钟时间及区间 TPOT。
若32请求轮换始终打断 decode，可另外用8个同时提交的长输出请求，选择全部完成prefill后、首请求结束前的公共区间；
该实验标为 decode 诊断，不与32请求端到端点混合。

FPM 可标记 prefill/decode 和 batch，但 `wall_time` 在本仓库优先来自 DeviceTimer，可能排除 Python 调度间隙，
且不同 speculative 路径的计时边界需要核实。不要简单汇总 FPM `wall_time` 或倒算TPOT当作完整服务的稳态TPS。
现有 native bench 的 `decode_window_tps` 保持 null；缺乏区间证据时如实保留未知。
