import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.speculative_hook import (
    _handle_lagom_mode,
    _resolve_lagom_window,
)
from sglang.srt.managers.hisparse_prefetcher import create_hisparse_prefetcher
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
