"""Actual multi-stream capture/replay against eager reference attention.

No checkpoint is required. Run this before the model-serving smoke benchmark.
"""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_graph import OasisKVGraphState
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "native,heads,group,kd,vd",
    [
        (False, 2, 1, 64, 64),
        (False, 2, 4, 128, 128),
        (False, 1, 4, 576, 512),
        (True, 1, 4, 576, 512),
    ],
)
@pytest.mark.parametrize("paired", [False, True])
def test_capture_repeated_replay_reorder_padding_growth_and_reuse(
    native, heads, group, kd, vd, paired
):
    torch.manual_seed(117)
    width = 2 if paired else 1
    config = OasisKVConfig(
        token_budget=8 if native else 32,
        block_size=1 if native else 4,
        native_sparse=native,
        max_new_blocks=1,
    )
    cache = OasisKVCache(config, "cuda")
    reference_cache = OasisKVCache(config, "cpu")
    graph_state = OasisKVGraphState(cache, 2, 96, 2)
    layers = [
        SimpleNamespace(layer_id=i, scaling=kd**-0.5, logit_cap=0) for i in range(2)
    ]
    inputs = [
        (
            torch.zeros(
                2 * width, heads * group, kd, dtype=torch.bfloat16, device="cuda"
            ),
            torch.zeros(2 * width, heads, kd, dtype=torch.bfloat16, device="cuda"),
            torch.zeros(2 * width, heads, vd, dtype=torch.bfloat16, device="cuda"),
            torch.full(
                (2 * width, config.history_blocks), -1, dtype=torch.int32, device="cuda"
            ),
        )
        for _ in layers
    ]

    def run():
        outputs = [
            graph_state.forward(
                q, k, v, layer, 2, paired=paired, topk=topk if native else None
            )
            for layer, (q, k, v, topk) in zip(layers, inputs)
        ]
        if paired:
            graph_state.finish_in_graph()
        return outputs

    # Warmup and capture run inactive rows, never fake cache requests.
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        outputs = run()
    assert not cache.layers
    table_address = graph_state.tables.data_ptr()

    def prefill(slot, generation, length):
        for layer in layers:
            key = torch.randn(length, heads, kd, dtype=torch.bfloat16)
            value = torch.randn(length, heads, vd, dtype=torch.bfloat16)
            reference_cache.get(slot, generation, layer.layer_id, key, value).append(
                key, value, 0
            )
            gpu_k, gpu_v = key.cuda(), value.cuda()
            cache.get(slot, generation, layer.layer_id, gpu_k, gpu_v).append(
                gpu_k, gpu_v, 0
            )

    prefill(1, 0, 15)
    prefill(2, 0, 16)
    schedules = [
        ([1, 2], [0, 0]),
        ([2], [0]),
        ([2, 1], [0, 0]),
        ([1], [1]),
        ([1, 2], [1, 0]),
    ]
    for step, (slots, generations) in enumerate(schedules):
        if step == 3:
            cache.release(1)
            reference_cache.release(1)
            prefill(1, 1, 7)
        prefixes = [
            reference_cache.layers[(s, g, 0)].length for s, g in zip(slots, generations)
        ]
        expected = []
        for layer, (q, k, v, topk) in zip(layers, inputs):
            q.normal_()
            k.normal_()
            v.normal_()
            # Force real selections away from the previous prediction: refill
            # must remain correct with misses and a one-block prefetch cap.
            topk[:, :] = torch.arange(config.history_blocks, device="cuda")
            if paired:
                topk[1::2] += 4
            layer_expected = []
            for b, (slot, generation, prefix) in enumerate(
                zip(slots, generations, prefixes)
            ):
                state = reference_cache.layers[(slot, generation, layer.layer_id)]
                start = b * width
                qs, ks, vs = (
                    q[start : start + width].cpu(),
                    k[start : start + width].cpu(),
                    v[start : start + width].cpu(),
                )
                state.append(ks[:1], vs[:1], prefix)
                state.prepare_attention(qs[0], topk[start].cpu() if native else None)
                layer_expected.append(
                    state.attend(
                        qs,
                        layer.scaling,
                        draft_key=ks[1] if paired else None,
                        draft_value=vs[1] if paired else None,
                    )
                )
                if paired:
                    state.prefetch(qs[1], topk[start + 1].cpu() if native else None)
            expected.append(torch.cat(layer_expected).flatten(1))
        graph_state.prepare(slots, generations, prefixes, paired=paired)
        cuda_graph.replay()
        graph_state.finish_replay()
        torch.cuda.synchronize()
        for layer, actual, wanted in zip(layers, outputs, expected):
            torch.testing.assert_close(
                actual[: len(slots) * width].cpu(), wanted, atol=2e-2, rtol=2e-2
            )
            assert not actual[len(slots) * width :].any()
            for slot, generation in zip(slots, generations):
                got = cache.layers[(slot, generation, layer.layer_id)]
                ref = reference_cache.layers[(slot, generation, layer.layer_id)]
                assert got.length == ref.length
                torch.testing.assert_close(got.resident.cpu(), ref.resident)
                torch.testing.assert_close(got.ages.cpu(), ref.ages)
        assert table_address == graph_state.tables.data_ptr()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
