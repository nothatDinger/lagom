"""Trace CLI -> real V4 memory sizing -> real state allocation, without weights.

Allocate small CPU pools, but keep the production configurator, KV pool and
compressor state constructors. Parsing a thirteen-row command alone did not catch
the old sixteen-slot C4 ring's ten-row startup limit.
"""

import argparse
from types import SimpleNamespace as NS

import pytest
import torch
from test_oasiskv_trace import runner

from sglang.srt.arg_groups.overrides import _deepseek_v4_overrides
from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.environ import envs
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_configurator(server_args):
    return DSV4PoolConfigurator(
        NS(
            kv_cache_dtype_str="fp8_e4m3",
            model_config=NS(
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                index_head_dim=128,
                context_len=server_args.context_length,
                compress_ratios=[4, 128],
                window_size=256,
            ),
            layer_info=NS(start_layer=0, end_layer=2),
            ps=NS(pp_size=1, attn_dp_size=1),
            server_args=server_args,
            is_draft_worker=False,
        )
    )


@pytest.mark.parametrize("width", [6, 7])
def test_matrix_command_sizes_and_allocates_thirteen_row_compressor(width, tmp_path):
    args = runner.build_parser().parse_args(
        ["--mode", "paired", "--verify-widths", "6", "7", "--output", str(tmp_path)]
    )
    args.top_k = 512
    run = runner.resolve_runs(args, 5)[width - 6]
    command = runner.server_command(run, 5)
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(command[3:])
    server_args = ServerArgs(model_path="dummy")
    for key, value in vars(parsed).items():
        setattr(server_args, key, value)
    server_args.device = "cuda"
    _handle_dspark(server_args)
    # launch_server resolves V4's paged SWA layout before pool configuration.
    for key, value in _deepseek_v4_overrides(
        server_args, NS(architectures=["DeepseekV4ForCausalLM"])
    ).items():
        setattr(server_args, key, value)
    assert server_args.page_size == 256
    assert server_args.max_speculative_num_draft_tokens == 13
    assert run.resident_units == 4096

    with (
        envs.SGLANG_OPT_USE_ONLINE_COMPRESS.override(False),
        get_context().override_server_args(
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=13,
            speculative_adaptive=False,
        ),
    ):
        config = make_configurator(server_args)
        assert (config.c4_ring_size, config.c128_ring_size) == (32, 256)
        # Keep the safety guard: an undersized pool must still reject G=12.
        config.c4_ring_size = 16
        with pytest.raises(AssertionError, match="serves at most 10"):
            config._assert_ring_serves_draft_tokens(13)
        config.c4_ring_size = 32

        sizes = config.calculate_pool_sizes_from_max_tokens(8192, 256)
        sizes.max_running_requests = 1
        config.finalize_with_max_running_requests(sizes)
        assert sizes.c4_state_pool_size == sizes.swa_max_total_num_tokens // 256 * 32

        # A thirteen-row ring costs more state memory; profiling must account for
        # it too, rather than only enlarging the physical allocation.
        server_args.__dict__.pop("max_speculative_num_draft_tokens")
        server_args.speculative_num_draft_tokens = 10
        smaller = make_configurator(server_args)
        assert smaller.c4_ring_size == 16
        assert config.bytes_per_full_token > smaller.bytes_per_full_token
        assert (
            config.calculate_pool_sizes(64 << 20, 256).max_total_num_tokens
            < smaller.calculate_pool_sizes(64 << 20, 256).max_total_num_tokens
        )

        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=1,
            swa_size=sizes.swa_max_total_num_tokens,
            c4_size=sizes.c4_max_total_num_tokens,
            c128_size=sizes.c128_max_total_num_tokens,
            c4_state_pool_size=sizes.c4_state_pool_size,
            c128_state_pool_size=sizes.c128_state_pool_size,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[4, 128],
            enable_hisparse=True,
        )
        for state in (
            pool.compress_state_pools[0],
            pool.indexer_compress_state_pools[0],
        ):
            assert state.ring_size == 32
            assert state.kv_score_buffer.shape[0] >= sizes.c4_state_pool_size + 32
            # Last SWA page, including a wrap in the ring; all thirteen writes
            # stay distinct and in bounds for both attention and indexer state.
            for offset in range(32):
                loc = sizes.swa_max_total_num_tokens - 256 + 64 + offset
                slots = state.translate_from_swa_loc_to_state_loc(
                    torch.arange(loc, loc + 13)
                )
                assert slots.unique().numel() == 13
                assert slots.max() < state.kv_score_buffer.shape[0] - 1
        assert pool.compress_state_pools[1].ring_size == 256
        # An adaptive tier/temporary width override cannot change allocated
        # state strides. Plans and reset/transfer paths read get_ring_size().
        with get_context().override_server_args(speculative_num_draft_tokens=6):
            assert pool.get_ring_size(4) == 32
            assert pool.get_ring_size(128) == 256
