import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.speculative_hook import (
    _handle_lagom_mode,
    _resolve_lagom_window,
)
from sglang.srt.managers.hisparse_prefetcher import create_hisparse_prefetcher
from sglang.srt.mem_cache.hisparse_spec import validate_pr_fused_config
from sglang.srt.runtime_context import get_context, max_speculative_num_draft_tokens
from sglang.srt.server_args import ServerArgs


def _args(**updates):
    values = dict(
        hisparse_config='{"prefetcher":"lagom"}',
        enable_hisparse=True,
        speculative_algorithm=None,
        speculative_num_draft_tokens=8,
        speculative_adaptive=False,
        device="cuda",
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        disaggregation_mode="null",
        is_lagom=False,
        lagom_prefetch_tokens=0,
        disable_cuda_graph=False,
        get_model_config=lambda: SimpleNamespace(hf_config=object()),
    )
    values.update(updates)
    return SimpleNamespace(**values)


def _select(args):
    with (
        patch("sglang.srt.configs.model_config.is_deepseek_v4", return_value=True),
        patch("sglang.srt.utils.is_hip", return_value=False),
        patch.dict("os.environ", {}, clear=True),
    ):
        _handle_lagom_mode(args)
    _resolve_lagom_window(args)
    return args


def test_lagom_default_prediction_window_preserves_decode_and_graphs():
    args = _select(_args())
    assert args.is_lagom
    assert args.speculative_algorithm == "DSPARK"
    assert args.speculative_num_draft_tokens == 8
    assert args.lagom_prefetch_tokens == 7
    assert (
        json.loads(args.hisparse_config)["prefetcher_config"]["prediction_steps"] == 7
    )
    assert not args.disable_cuda_graph


def test_lagom_explicit_prediction_window():
    args = _select(
        _args(
            hisparse_config=json.dumps(
                {"prefetcher": "lagom", "prefetcher_config": {"prediction_steps": 3}}
            )
        )
    )
    assert args.speculative_num_draft_tokens == 8
    assert args.lagom_prefetch_tokens == 3


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "3"])
def test_lagom_rejects_invalid_prediction_window(value):
    with pytest.raises(ValueError, match="positive integer"):
        _select(
            _args(
                hisparse_config=json.dumps(
                    {
                        "prefetcher": "lagom",
                        "prefetcher_config": {"prediction_steps": value},
                    }
                )
            )
        )


@pytest.mark.parametrize(
    "update,match",
    [
        ({"speculative_algorithm": "EAGLE3"}, "requires DSPARK"),
        ({"enable_hisparse": False}, "enable-hisparse"),
        ({"dp_size": 2}, "tensor parallelism only"),
        ({"attn_cp_size": 2}, "tensor parallelism only"),
        ({"device": "cpu"}, "CUDA DeepSeek-V4"),
        (
            {"hisparse_config": '{"prefetcher":"lagom","force_h2d_each_step":true}'},
            "preserve prefetched KV",
        ),
    ],
)
def test_lagom_rejects_incompatible_modes(update, match):
    with pytest.raises(ValueError, match=match):
        _select(_args(**update))


def test_lagom_allocation_bound_includes_probes_but_public_width_does_not():
    args = ServerArgs(
        model_path="dummy",
        speculative_num_draft_tokens=8,
        is_lagom=True,
        lagom_prefetch_tokens=7,
    )
    assert args.max_speculative_num_draft_tokens == 15
    override = get_context().override_server_args(
        speculative_num_draft_tokens=8, is_lagom=True, lagom_prefetch_tokens=7
    )
    override.install()
    try:
        assert max_speculative_num_draft_tokens() == 15
    finally:
        override.restore()
    assert args.speculative_num_draft_tokens == 8


def test_lagom_grows_c4_ring_without_aliasing_previous_compressor_window():
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        get_compress_state_ring_size,
        get_compress_state_write_pad,
    )

    assert get_compress_state_ring_size(4, True) == 16
    ring = get_compress_state_ring_size(4, True, 15)
    assert ring == 32
    assert 128 % ring == 0
    assert get_compress_state_write_pad(4, ring) >= 15
    assert get_compress_state_ring_size(4, True, 122) == 128
    with pytest.raises(ValueError, match="at most 122"):
        get_compress_state_ring_size(4, True, 123)


def test_lagom_prefetch_config_counts_each_probe_in_c4_entries():
    prefetcher = create_hisparse_prefetcher(
        "lagom",
        {"prediction_steps": 7, "size": 256},
        effective_top_k=512,
        device_buffer_size=8192,
    )
    assert prefetcher.prediction_steps == 7
    assert prefetcher.logical_entries == 256
    with pytest.raises(ValueError, match="model Top-K"):
        create_hisparse_prefetcher(
            "lagom",
            {"size": 1024},
            effective_top_k=512,
            device_buffer_size=8192,
        )


@pytest.mark.parametrize("decode_width,prediction_steps", [(8, 7), (2, 9), (8, 1)])
def test_lagom_capacity_protects_real_and_prediction_union(
    decode_width, prediction_steps
):
    width, top_k = decode_width + prediction_steps, 512
    config = dict(
        cuda=True,
        dsv4=True,
        speculative=True,
        width=width,
        top_k=top_k,
        hot=width * top_k,
        page_size=64,
        force_h2d=False,
        skip_io=False,
        prefetcher="lagom",
        prediction_steps=prediction_steps,
    )
    validate_pr_fused_config(**config)
    # The previous sequential implementation could accept only the larger
    # window's capacity. Concurrent transfers need the complete protected union.
    config["hot"] = max(decode_width, prediction_steps) * top_k
    with pytest.raises(ValueError, match="full verify union"):
        validate_pr_fused_config(**config)


def test_lagom_combined_window_respects_kernel_occurrence_limit():
    config = dict(
        cuda=True,
        dsv4=True,
        speculative=True,
        width=16,
        top_k=512,
        hot=16384,
        page_size=64,
        force_h2d=False,
        skip_io=False,
        prefetcher="lagom",
        prediction_steps=8,
    )
    validate_pr_fused_config(**config)
    config["width"] = 17
    with pytest.raises(ValueError, match="full verify union"):
        validate_pr_fused_config(**config)


@pytest.mark.parametrize("device_buffer_size", [None, 8192])
def test_lagom_pool_budget_uses_complete_planning_union(monkeypatch, device_buffer_size):
    from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator

    raw_config = {"prefetcher": "lagom"}
    if device_buffer_size is not None:
        raw_config["device_buffer_size"] = device_buffer_size
    args = _args(
        speculative_algorithm="DSPARK",
        is_lagom=True,
        lagom_prefetch_tokens=7,
        hisparse_config=json.dumps(raw_config),
        max_speculative_num_draft_tokens=15,
        swa_full_tokens_ratio=1,
        max_running_requests=4,
        disaggregation_decode_extra_slots=0,
    )
    model_config = SimpleNamespace(
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        context_len=65536,
        compress_ratios=[4, 128],
        window_size=128,
        hf_text_config=SimpleNamespace(index_topk=512),
    )
    kvc = SimpleNamespace(
        kv_cache_dtype_str="fp8_e4m3",
        model_config=model_config,
        layer_info=SimpleNamespace(start_layer=0, end_layer=2),
        ps=SimpleNamespace(pp_size=1, attn_dp_size=1),
        server_args=args,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.pool_configurator.is_hip", lambda: False
    )
    monkeypatch.setattr(
        DSV4PoolConfigurator, "_get_bytes_per_full_token", lambda _: 1.0
    )
    with patch.dict("os.environ", {"SGLANG_OPT_USE_ONLINE_COMPRESS": "0"}):
        pool = DSV4PoolConfigurator(kvc)
    assert pool.hisparse_spec_layout.verify_width == 15
    assert pool.hisparse_spec_layout.occurrences == 15 * 512
    # Retain the previous 2 * max(W+1, P) * K default while protecting the
    # entire W+P+1 union in the planner and memory estimate.
    assert pool.hisparse_spec_layout.hot_slots == 8192
    # The request pool reserves a padding slot in addition to the four live
    # requests. Include all stable per-layer transfer tensors in the budget.
    layout = pool.hisparse_spec_layout
    expected = layout.fixed_state_bytes(num_layers=1, num_req_slots=5)
    expected += layout.lagom_plan_bytes(num_layers=1, num_req_slots=5)
    assert pool._get_spec_swapin_fixed_bytes_for_token_capacity(65536) == expected


def test_lagom_transfer_plan_budget_matches_tensor_storage():
    import torch

    from sglang.srt.mem_cache.hisparse_spec import HiSparseSpecLayout

    layout = HiSparseSpecLayout(8192, 15, 512)
    requests, layers = 5, 2
    tensors = (
        torch.empty((layers, requests, layout.occurrences), dtype=torch.int64),
        torch.empty((layers, requests, layout.occurrences), dtype=torch.int32),
        torch.empty((layers, requests, layout.occurrences), dtype=torch.int32),
        torch.empty((layers, requests), dtype=torch.int32),
        torch.empty((layers,), dtype=torch.int32),
    )
    assert layout.lagom_plan_bytes(num_layers=layers, num_req_slots=requests) == sum(
        value.untyped_storage().nbytes() for value in tensors
    )
