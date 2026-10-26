"""Exercise the real paired capture plumbing and LM-head row gather.

CPU checks ownership after capture locals are destroyed. CUDA additionally
replays all buckets under allocation churn and changing request-pool slots.
No model checkpoint or attention kernel is required for this regression.
"""

import contextlib
import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _runner(device, monkeypatch, algorithm):
    import sglang.srt.model_executor.runner.decode_cuda_graph_runner as module

    for name in (
        "set_dp_buffer_len",
        "set_is_extend_in_batch",
        "maybe_flashinfer_autotune_speculative_draft",
    ):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "enable_num_token_non_padded", lambda: False)
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.captured_req_width = 2
    runner.speculative_num_draft_tokens = 2
    runner.speculative_num_steps = 1
    runner.ragged_verify_mode = False
    runner.capture_forward_mode = ForwardMode.TARGET_VERIFY
    runner.capture_hidden_mode = CaptureHiddenMode.FULL
    runner.is_encoder_decoder = runner.require_gathered_buffer = (
        runner.enable_prefill_cp
    ) = False
    runner.require_mlp_tp_gather = runner.require_attn_tp_gather = False
    runner.enable_profile_cuda_graph = False
    runner.pp_size = 1
    runner._record_in_graph_metadata_prep_done = lambda: None
    runner.attn_backend = SimpleNamespace(
        init_forward_metadata_out_graph=lambda *a, **k: None,
        init_forward_metadata_in_graph=lambda *a: None,
        finish_forward_in_graph=lambda *a: None,
    )
    runner.model_runner = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_lora=False,
            is_oasiskv_lookahead=True,
            debug_cuda_graph=False,
            speculative_eagle_topk=1,
        ),
        device=device,
        dtype=torch.float32,
        model_config=SimpleNamespace(hidden_size=4),
        is_draft_worker=False,
        hisparse_coordinator=None,
        spec_algorithm=algorithm,
        tp_group=SimpleNamespace(barrier=lambda: None),
        model=object(),
        attn_backend=runner.attn_backend,
        canary_manager=None,
        capture_tail_hooks=[],
    )
    runner.tbo_plugin = SimpleNamespace(capture_one_batch_size=lambda *a, **k: None)
    runner.deepep_adapter = SimpleNamespace(capture=lambda **k: None)
    runner.device_module = (
        torch.cuda
        if device == "cuda"
        else SimpleNamespace(
            synchronize=lambda: None, graph=lambda **k: contextlib.nullcontext()
        )
    )
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    runner.buffers = DecodeInputBuffers.create(
        device=torch.device(device),
        max_bs=8,
        max_num_token=16,
        hidden_size=4,
        next_token_logits_buffer=torch.zeros(16, 4, device=device),
        dtype=torch.float32,
        dp_size=1,
        pp_size=1,
        is_encoder_decoder=False,
        require_mlp_tp_gather=False,
        seq_len_fill_value=1,
        encoder_len_fill_value=0,
        num_tokens_per_req=2,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=False,
    )

    def slot(name):
        value = getattr(runner.buffers, name)

        def slice_for(bs, tokens):
            if name == "mrope_positions":
                return value[:, :tokens]
            return value[
                : tokens if name in ("input_ids", "positions", "out_cache_loc") else bs
            ]

        return SimpleNamespace(slice_for=slice_for)

    runner.buffer_registry = SimpleNamespace(get_slot=slot, has_slot=lambda _: False)
    runner.backend = FullCudaGraphBackend(runner)
    if device == "cuda":
        runner.backend._pool = torch.cuda.graph_pool_handle()
        runner.backend._capture_stream = torch.cuda.Stream()
    return runner


def test_dspark_draft_capture_uses_one_row_without_target_probe_metadata(monkeypatch):
    runner = _runner("cpu", monkeypatch, SpeculativeAlgorithm.DSPARK)
    runner.model_runner.is_draft_worker = True
    runner.captured_req_width = (
        SpeculativeAlgorithm.DSPARK.get_num_tokens_per_req_for_target_verify(
            num_draft_tokens=2, is_draft_worker=True
        )
    )
    assert runner.captured_req_width == 1
    batch, _, _ = runner.capture_prepare(2, num_tokens=2)
    assert batch.spec_info.draft_token_num == 1
    assert not batch.is_oasiskv_paired
    assert batch.input_ids.numel() == batch.batch_size == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "algorithm", [SpeculativeAlgorithm.EAGLE3, SpeculativeAlgorithm.DSPARK]
)
def test_capture_keeps_pair_indices_for_each_bucket_after_allocations_and_slot_reuse(
    device, algorithm, monkeypatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for actual replay")
    runner = _runner(device, monkeypatch, algorithm)
    embedding = torch.arange(4096 * 4, dtype=torch.float32, device=device).view(4096, 4)
    refs = {}

    def observe(_runner, _output, batch, _tokens):
        refs[batch.batch_size] = (
            weakref.ref(batch),
            weakref.ref(batch.oasiskv_normal_rows._base),
            weakref.ref(batch.oasiskv_draft_valid),
        )

    runner.model_runner.capture_tail_hooks.append(observe)

    def forward(ids, positions, batch):
        hidden = embedding[ids] + batch.req_pool_indices.repeat_interleave(2)[:, None]
        # Use the production LogitsMetadata and the actual failing gather path.
        metadata = LogitsMetadata.from_forward_batch(batch)
        normal = LogitsProcessor._get_pruned_states(None, hidden, None, None, metadata)[
            0
        ]
        return (
            normal,
            hidden[batch.oasiskv_draft_rows],
            batch.oasiskv_draft_valid.clone(),
        )

    for bs in (8, 4, 2, 1):
        runner.capture_one_shape(bs, forward)
    gc.collect()
    for bs, owners in refs.items():
        assert all(
            ref() is not None for ref in owners
        ), f"bs={bs}: capture input owner lost"
        assert owners[1]().tolist() == list(range(2 * bs))

    # Pool slots deliberately exceed 2B. They are dynamic request identities,
    # never paired-row indices. Return to bs=1 repeatedly after other buckets.
    for step, (bs, slots) in enumerate(
        [
            (1, [0]),
            (1, [511]),
            (2, [1023, 7]),
            (1, [7]),
            (4, [511, 1023, 7]),
            (8, [7]),
            (1, [1023]),
            (2, [7, 511]),
        ]
    ):
        raw_bs = len(slots)
        live = SimpleNamespace(
            input_ids=torch.arange(raw_bs * 2, device=device) + step * 19,
            req_pool_indices=torch.tensor(slots, device=device),
            seq_lens=torch.full((raw_bs,), 16384 + step, device=device),
            seq_lens_cpu=torch.full((raw_bs,), 16384 + step),
            out_cache_loc=torch.arange(raw_bs * 2, device=device),
            positions=torch.arange(raw_bs * 2, device=device),
            mrope_positions=None,
        )
        runner.buffers.populate_from_forward_batch(
            forward_batch=live,
            raw_bs=raw_bs,
            raw_num_token=raw_bs * 2,
            bs=bs,
            seq_len_fill_value=1,
            require_gathered_buffer=False,
            num_tokens_per_req=2,
            dsa_enable_prefill_cp=False,
            enable_num_token_non_padded_flag=False,
        )
        # Reuse the allocation size classes of the capture-only row/mask tensors.
        churn = [
            torch.full((n,), 10_000 + step, dtype=torch.int64, device=device)
            for n in (2, 4, 8, 16)
            for _ in range(16)
        ]
        churn += [
            torch.zeros(n, dtype=torch.bool, device=device)
            for n in (1, 2, 4, 8)
            for _ in range(16)
        ]
        key = ShapeKey(size=bs)
        if device == "cuda":
            actual = runner.backend.replay(key, live)
            torch.cuda.synchronize()
        else:
            # CPU cannot replay a CUDA graph, but exercises updated inputs using
            # exactly the retained capture batch whose pointers replay consumes.
            batch = runner.backend._capture_inputs[key]
            actual = forward(batch.input_ids, batch.positions, batch)
            del batch
        expected = (
            embedding[live.input_ids]
            + live.req_pool_indices.repeat_interleave(2)[:, None]
        )
        torch.testing.assert_close(actual[0][:raw_bs], expected[::2])
        torch.testing.assert_close(actual[1][:raw_bs], expected[1::2])
        assert actual[2].all()
        assert all(ref() is not None for owners in refs.values() for ref in owners)
        del churn
    runner.backend.cleanup()
    gc.collect()
    assert all(ref() is None for owners in refs.values() for ref in owners)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
