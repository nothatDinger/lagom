"""Exercise the real backend with scheduler-shaped batches and projected KV."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.arg_groups.speculative_hook import _handle_oasiskv_lookahead
from sglang.srt.layers.attention.oasiskv_backend import OasisKVAttnBackend
from sglang.srt.mem_cache.oasiskv_cache import OasisKVCache
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.speculative.oasiskv_lookahead import (
    build_oasiskv_paired_batch,
    configure_oasiskv_forward_batch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def batch(slots, prefixes, lengths, *, paired=False):
    mode = SimpleNamespace(is_idle=lambda: False, is_decode=lambda: False)
    fb = SimpleNamespace(
        req_pool_indices_cpu=torch.tensor(slots),
        batch_size=len(slots),
        forward_mode=mode,
        seq_lens_cpu=torch.tensor(prefixes) + torch.tensor(lengths),
        extend_prefix_lens_cpu=prefixes,
        extend_seq_lens_cpu=lengths,
    )
    if paired:
        pair = build_oasiskv_paired_batch(
            torch.arange(len(slots)),
            torch.arange(len(slots)) + 10,
            torch.tensor(prefixes),
        )
        fb.input_ids = pair.input_ids
        fb.out_cache_loc = torch.arange(sum(lengths))
        fb.extend_prefix_lens = torch.tensor(prefixes)
        fb.extend_start_loc = torch.tensor([0, 2])[: len(slots)]
        fb.seq_lens_cpu = torch.tensor(prefixes)
        fb.seq_lens = fb.seq_lens_cpu.clone()
        configure_oasiskv_forward_batch(fb, pair)
    return fb


@pytest.mark.parametrize(
    "heads,group,kdim,vdim,rope", [(2, 1, 8, 8, 0), (2, 3, 8, 8, 0), (1, 4, 12, 8, 4)]
)
def test_backend_multirequest_prefill_pair_and_committed_history(
    heads, group, kdim, vdim, rope
):
    torch.manual_seed(91)
    cache = OasisKVCache(OasisKVConfig(token_budget=16, block_size=4), "cpu")
    runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_generation=torch.tensor([0, 3, 4])),
        token_to_kv_pool=SimpleNamespace(oasiskv_cache=cache),
    )
    backend = OasisKVAttnBackend(runner)
    layer = SimpleNamespace(
        tp_q_head_num=heads * group,
        tp_k_head_num=heads,
        tp_v_head_num=heads,
        scaling=kdim**-0.5,
        logit_cap=0,
        is_cross_attention=False,
        layer_id=0,
    )
    keys = [torch.randn(8, heads, kdim) for _ in range(2)]
    values = [torch.randn(8, heads, vdim) for _ in range(2)]
    queries = [torch.randn(8, heads * group, kdim) for _ in range(2)]

    def call(fb, qs, ks, vs):
        backend.init_forward_metadata(fb)
        kwargs = {}
        if rope:
            kwargs = dict(q_rope=qs[..., -rope:], k_rope=ks[..., -rope:])
            qs, ks = qs[..., :-rope], ks[..., :-rope]
        return backend.forward_extend(qs, ks, vs, layer, fb, **kwargs).view(
            -1, heads * group, vdim
        )

    lengths = [3, 5]
    call(
        batch([1, 2], [0, 0], lengths),
        torch.cat([q[:n] for q, n in zip(queries, lengths)]),
        torch.cat([k[:n] for k, n in zip(keys, lengths)]),
        torch.cat([v[:n] for v, n in zip(values, lengths)]),
    )
    for offset in range(2):
        prefixes = [n + offset for n in lengths]
        result = call(
            batch([1, 2], prefixes, [2, 2], paired=True),
            torch.cat([q[p : p + 2] for q, p in zip(queries, prefixes)]),
            torch.cat([k[p : p + 2] for k, p in zip(keys, prefixes)]),
            torch.cat([v[p : p + 2] for v, p in zip(values, prefixes)]),
        )
        for request, p in enumerate(prefixes):
            mask = torch.arange(p + 2)[None] <= p + torch.arange(2)[:, None]
            expected = F.scaled_dot_product_attention(
                queries[request][p : p + 2].transpose(0, 1),
                keys[request][: p + 2].repeat_interleave(group, 1).transpose(0, 1),
                values[request][: p + 2].repeat_interleave(group, 1).transpose(0, 1),
                attn_mask=mask,
                scale=layer.scaling,
            ).transpose(0, 1)
            torch.testing.assert_close(result[request * 2 : request * 2 + 2], expected)
            state = cache.layers[(request + 1, request + 3, 0)]
            assert state.length == p + 1
            assert state.prefetch_submissions == offset + 1


def test_native_indexer_metadata_stays_immutable_and_logical():
    @dataclass(frozen=True)
    class Metadata:
        force_unfused_topk: bool = False

    original = Metadata()
    backend = OasisKVAttnBackend.__new__(OasisKVAttnBackend)
    backend.dsa = SimpleNamespace(get_indexer_metadata=lambda *args: original)
    result = backend.get_indexer_metadata(0, None)
    assert result.force_unfused_topk
    assert not original.force_unfused_topk


def test_native_backend_repairs_wrong_draft_prediction_with_real_topk():
    cache = OasisKVCache(
        OasisKVConfig(
            token_budget=2, block_size=1, max_new_blocks=1, native_sparse=True
        ),
        "cpu",
    )
    backend = OasisKVAttnBackend.__new__(OasisKVAttnBackend)
    backend.cache, backend.dsa = cache, None
    backend.req_pool = SimpleNamespace(req_generation=torch.tensor([0, 1]))
    layer = SimpleNamespace(
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        scaling=0.5,
        logit_cap=0,
        is_cross_attention=False,
        layer_id=0,
    )
    keys, values, queries = (
        torch.randn(7, 1, 4),
        torch.randn(7, 1, 3),
        torch.randn(7, 2, 4),
    )
    fb = batch([1], [0], [4])
    backend.init_forward_metadata(fb)
    backend.forward_extend(
        queries[:4],
        keys[:4],
        values[:4],
        layer,
        fb,
        topk_indices=torch.tensor([[0, -1], [0, 1], [1, 2], [0, 3]]),
    )
    for prefix, root_topk, draft_topk, history in (
        (4, [0, 1], [2, 3], [0, 1]),
        (5, [0, 3], [0, 3], [0, 3]),
    ):
        fb = batch([1], [prefix], [2], paired=True)
        backend.init_forward_metadata(fb)
        output = backend.forward_extend(
            queries[prefix : prefix + 2],
            keys[prefix : prefix + 2],
            values[prefix : prefix + 2],
            layer,
            fb,
            topk_indices=torch.tensor([root_topk, draft_topk]),
        ).view(2, 2, 3)
        selected = history + [prefix]
        expected = F.scaled_dot_product_attention(
            queries[prefix].unsqueeze(1),
            keys[selected].transpose(0, 1),
            values[selected].transpose(0, 1),
            scale=0.5,
        ).squeeze(1)
        torch.testing.assert_close(output[0], expected)
        assert cache.layers[(1, 1, 0)].length == prefix + 1


def test_idle_backend_does_not_append_or_allocate_requests():
    cache = OasisKVCache(OasisKVConfig(), "cpu")
    backend = OasisKVAttnBackend(
        SimpleNamespace(
            req_to_token_pool=None,
            token_to_kv_pool=SimpleNamespace(oasiskv_cache=cache),
        )
    )
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: True))
    backend.init_forward_metadata(fb)
    out = backend.forward_extend(
        torch.empty(0, 8),
        None,
        None,
        SimpleNamespace(tp_q_head_num=2, v_head_dim=4),
        fb,
    )
    assert out.shape == (0, 8) and not cache.layers


def test_repeated_autotune_dummy_batches_do_not_require_or_modify_request_history():
    cache = OasisKVCache(OasisKVConfig(), "cpu")
    backend = OasisKVAttnBackend(
        SimpleNamespace(
            req_to_token_pool=None,
            token_to_kv_pool=SimpleNamespace(oasiskv_cache=cache),
        )
    )
    fb = SimpleNamespace(
        is_oasiskv_dummy=True,
        batch_size=2,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )
    layer = SimpleNamespace(
        tp_q_head_num=4,
        tp_k_head_num=2,
        tp_v_head_num=2,
        is_cross_attention=False,
        scaling=0.5,
    )
    q, k, v = torch.randn(4, 4, 4), torch.randn(4, 2, 4), torch.randn(4, 2, 3)
    assert backend.get_cuda_graph_seq_len_fill_value() == 1
    for _ in range(2):
        backend.init_forward_metadata(fb)
        result = backend.forward_extend(q, k, v, layer, fb)
        assert result.shape == (4, 12) and torch.isfinite(result).all()
        assert not cache.layers


def generic_args(**overrides):
    args = dict(
        attention_backend="oasiskv",
        hisparse_config="{}",
        enable_hisparse=False,
        disable_radix_cache=True,
        pp_size=1,
        max_running_requests=None,
        speculative_draft_model_path="local-eagle3",
        speculative_algorithm=None,
        speculative_num_steps=None,
        speculative_eagle_topk=None,
        speculative_num_draft_tokens=None,
        kv_cache_dtype="auto",
        disable_cuda_graph=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_generic_cli_keeps_graphs_enabled_for_root_only_execution():
    args = generic_args()
    _handle_oasiskv_lookahead(args)
    assert args.is_oasiskv_lookahead and not args.disable_cuda_graph
    assert args.speculative_algorithm == "EAGLE3"
    assert args.speculative_num_draft_tokens == 2
    assert args.kv_cache_dtype == "bfloat16"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(enable_hisparse=True),
        dict(disable_radix_cache=False),
        dict(prefill_attention_backend="triton"),
        dict(attn_cp_size=2),
        dict(enable_two_batch_overlap=True),
    ],
)
def test_generic_cli_rejects_incompatible_cache_ownership(kwargs):
    with pytest.raises(ValueError):
        _handle_oasiskv_lookahead(generic_args(**kwargs))


@pytest.mark.parametrize("value", ["[]", "invalid"])
def test_generic_cli_rejects_malformed_configuration(value):
    with pytest.raises(ValueError, match="JSON object"):
        _handle_oasiskv_lookahead(generic_args(hisparse_config=value))
