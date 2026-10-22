from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.oasiskv_pool import (
    OasisKVDSATokenToKVPool,
    OasisKVTokenToKVPool,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("architecture", ["mha", "mla", "dsa"])
def test_runtime_pool_is_host_backed_and_release_follows_request_ownership(
    architecture,
):
    native = architecture == "dsa"
    override = get_context().override_server_args(
        attention_backend="oasiskv",
        enable_hisparse=False,
        enable_hierarchical_cache=False,
        enable_dsa_cache_layer_split=False,
        disaggregation_mode="null",
        max_running_requests=2,
        page_size=64 if native else 1,
        hisparse_config='{"top_k":16,"prefill_chunk_size":4}',
    )
    args = override.install()
    try:
        hf = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM" if native else "LlamaForCausalLM"]
        )
        if native:
            hf.index_topk, hf.index_head_dim = 16, 128
        hf.get_text_config = lambda: hf
        model = SimpleNamespace(
            hf_config=hf,
            hf_text_config=hf,
            context_len=128,
            head_dim=64,
            v_head_dim=64,
            num_attention_heads=4,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            linear_attn_registry_result=None,
            get_num_kv_heads=lambda *args: 2,
        )
        kvc = KVCacheConfigurator.__new__(KVCacheConfigurator)
        kvc.server_args, kvc.model_config = args, model
        kvc.is_draft_worker = kvc.is_hybrid_swa = False
        kvc.mambaish_config = None
        kvc.use_mla_backend = architecture != "mha"
        kvc.kv_cache_dtype, kvc.kv_cache_dtype_str = torch.bfloat16, "bfloat16"
        kvc.device = "cpu"
        kvc.layer_info = SimpleNamespace(
            start_layer=0, end_layer=2, num_effective_layers=2
        )
        kvc.spec_algorithm = SimpleNamespace(
            is_eagle=lambda: True,
            is_standalone=lambda: False,
            is_dflash_family=lambda: False,
        )
        kvc.spec_aux_config = SimpleNamespace(eagle_draft_num_layers=1)
        req_pool = ReqToTokenPool(2, 128, "cpu", False)
        with get_parallel().override(attn_tp_size=1, attn_cp_size=1, attn_dcp_size=1):
            pool = kvc._build_token_to_kv_pool(
                sizes=SimpleNamespace(max_total_num_tokens=128),
                is_dsa_model=native,
                is_dsv4_model=False,
                req_to_token_pool=req_pool,
            )
            assert isinstance(
                pool, OasisKVDSATokenToKVPool if native else OasisKVTokenToKVPool
            )
            assert req_pool.oasiskv_cache is pool.oasiskv_cache
            if native:
                assert all(t.shape == (1, 1, 576) for t in pool.kv_buffer)
                assert pool.index_key_cache.buffer[0].numel() > 128 * 128
            else:
                assert not hasattr(pool, "k_buffer") and not hasattr(pool, "kv_buffer")
            sizing = DefaultPoolConfigurator(kvc)
            # The draft remains a dense allocation even when target KV is offloaded.
            dense_target = sizing._compute_cell_size(kvc, 2)
            assert sizing._cell_size >= dense_target // 2
            budget = sizing._oasiskv_reserved_bytes + sizing._cell_size * 128
            assert (
                sizing.calculate_pool_sizes(
                    budget, kvc.pool_page_size
                ).max_total_num_tokens
                == 128
            )

        req = SimpleNamespace(req_pool_idx=None)
        slot = req_pool.alloc([req])[0]
        key = torch.randn(5, 1, 8)
        state = pool.oasiskv_cache.get(
            slot, int(req_pool.req_generation[slot]), 0, key, key
        )
        state.append(key, key, 0)
        req_pool.free(req)
        assert not pool.oasiskv_cache.layers
        assert req.req_pool_idx is None
        slot = req_pool.alloc([req])[0]
        assert req_pool.req_generation[slot] == 2
    finally:
        override.restore()
