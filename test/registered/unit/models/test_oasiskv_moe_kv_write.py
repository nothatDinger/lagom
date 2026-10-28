"""MoE KV-write dispatch through the real forward context, without model weights."""

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.oasiskv_backend import OasisKVAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_pool import (
    OasisKVDSATokenToKVPool,
    OasisKVTokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_req_to_token_pool,
    get_token_to_kv_pool,
)
from sglang.srt.models import qwen3_moe
from sglang.srt.models import utils as model_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_backend():
    req_pool = SimpleNamespace(size=2, req_generation=torch.tensor([0, 1]))
    pool = OasisKVTokenToKVPool(
        size=128,
        page_size=1,
        dtype=torch.bfloat16,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    pool.init_oasiskv(OasisKVConfig(token_budget=16, block_size=4), req_pool)
    backend = OasisKVAttnBackend(
        SimpleNamespace(req_to_token_pool=req_pool, token_to_kv_pool=pool)
    )
    return backend, req_pool, pool


def simulate_platform(monkeypatch, platform):
    monkeypatch.setattr(model_utils, "_is_cuda", platform == "cuda")
    monkeypatch.setattr(model_utils, "_is_hip", platform == "hip")
    monkeypatch.setattr(
        model_utils, "is_prefill_context_parallel_enabled", lambda: False
    )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("platform", ["cuda", "hip"])
def test_standard_pool_accessors_and_fused_write_opt_out(
    monkeypatch, wrapped, platform
):
    simulate_platform(monkeypatch, platform)
    backend, req_pool, pool = make_backend()
    if wrapped:
        backend = TboAttnBackend(backend, [])
    fb = SimpleNamespace(dcp_kv_mask=None)
    with forward_context(ForwardContext(attn_backend=backend)):
        assert get_token_to_kv_pool() is pool
        assert get_req_to_token_pool() is req_pool
        assert not model_utils.enable_fused_set_kv_buffer(fb)
        # The dense draft (or another model) must retain its own fusion policy.
        dense = SimpleNamespace(token_to_kv_pool=SimpleNamespace(dtype=torch.bfloat16))
        with forward_context(ForwardContext(attn_backend=dense)):
            assert model_utils.enable_fused_set_kv_buffer(fb)
            fb.dcp_kv_mask = torch.ones(1, dtype=torch.bool)
            assert not model_utils.enable_fused_set_kv_buffer(fb)
            fb.dcp_kv_mask = None
        assert not model_utils.enable_fused_set_kv_buffer(fb)


@pytest.mark.parametrize("platform", ["cuda", "hip"])
def test_native_dsa_pool_also_rejects_dense_fused_writes(monkeypatch, platform):
    simulate_platform(monkeypatch, platform)
    # No CUDA index-key allocation is needed to exercise this pool capability.
    pool = OasisKVDSATokenToKVPool.__new__(OasisKVDSATokenToKVPool)
    pool.dtype = torch.bfloat16
    with forward_context(
        ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=pool))
    ):
        assert not model_utils.enable_fused_set_kv_buffer(SimpleNamespace())


@pytest.mark.parametrize("phase", ["warmup", "prefill", "paired", "decode"])
def test_qwen3_moe_rope_and_attention_keep_backend_owned_kv_writes(monkeypatch, phase):
    simulate_platform(monkeypatch, "cuda")
    backend, _, pool = make_backend()
    # Execute the real Qwen3-MoE methods that gate RoPE/cache fusion and choose
    # save_kv_cache. Only QK normalization/RoPE arithmetic and the output linear
    # layer are replaced; RadixAttention and OasisKV perform the actual writes.
    attn = qwen3_moe.Qwen3MoeAttention.__new__(qwen3_moe.Qwen3MoeAttention)
    torch.nn.Module.__init__(attn)
    attn.head_dim, attn.q_size, attn.kv_size = 8, 64, 8
    attn.q_norm = attn.k_norm = torch.nn.Identity()
    attn.alt_stream = None
    attn.use_fused_qk_norm_rope = False
    attn.compatible_with_fused_kv_buffer = True
    attn.attn = RadixAttention(8, 8, 8**-0.5, 1, 0)
    attn.o_proj = lambda x: (x, None)
    monkeypatch.setattr(qwen3_moe, "apply_qk_norm", lambda q, k, **kwargs: (q, k))
    rope_calls = []

    def rope(positions, q, k, fused_set_kv_buffer_arg):
        rope_calls.append(fused_set_kv_buffer_arg)
        return q, k

    attn.rotary_emb = rope
    prefix = 3 if phase in ("paired", "decode") else 0
    width = 1 if phase == "decode" else 2
    fb = SimpleNamespace(
        forward_mode={
            "decode": ForwardMode.DECODE,
            "paired": ForwardMode.TARGET_VERIFY,
        }.get(phase, ForwardMode.EXTEND),
        is_oasiskv_dummy=phase == "warmup",
        is_oasiskv_paired=phase == "paired",
        batch_size=1,
        req_pool_indices_cpu=torch.tensor([1]),
        seq_lens_cpu=torch.tensor([prefix if phase == "paired" else prefix + width]),
        extend_prefix_lens_cpu=[prefix],
        extend_seq_lens_cpu=[width],
    )
    if prefix:
        history = torch.randn(prefix, 1, 8, dtype=torch.bfloat16)
        pool.oasiskv_cache.get(1, 1, 0, history, history).append(history, history, 0)
    qkv = torch.randn(width, 80, dtype=torch.bfloat16)
    backend.init_forward_metadata(fb)
    with forward_context(ForwardContext(attn_backend=backend)):
        q, k, v = attn.apply_qk_norm_rope(qkv, torch.arange(prefix, prefix + width), fb)
        output = attn.forward_core((None, fb, (q, k, v, fb)))
    assert rope_calls == [None]
    assert output.shape == (width, 64) and torch.isfinite(output).all()
    if phase == "warmup":
        assert not pool.oasiskv_cache.layers
    else:
        state = pool.oasiskv_cache.layers[(1, 1, 0)]
        committed = 1 if phase == "paired" else width
        assert state.length == prefix + committed
        expected = torch.cat((k[:committed], v[:committed]), -1).unsqueeze(0)
        torch.testing.assert_close(
            state.materialize_history()[:, prefix:, :16], expected
        )
