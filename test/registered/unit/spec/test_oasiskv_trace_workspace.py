"""Wide exact shadows preserve row KV without enlarging persistent residency."""

import json
from types import SimpleNamespace as NS

import pytest
import torch
from test_oasiskv_trace import (
    analyzer,
    auditor,
    fake_coordinator,
    records,
    runner,
    write_paired_run,
)

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.speculative import oasiskv_trace_workspace as workspace
from sglang.srt.speculative.oasiskv_trace import ForwardObservation
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def allocator(size=64, page_size=2):
    return PagedTokenToKVPoolAllocator(
        size=size,
        page_size=page_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=None,
        need_sort=True,
    )


def test_wide_exact_shadow_keeps_evicted_bytes_and_restores_allocator(monkeypatch):
    c = fake_coordinator()
    a = allocator()
    # Reserve slots holding the real hot cache; temporary copies must not alias.
    a.alloc(16)
    c.token_to_kv_pool_allocator = NS(hisparse_attn_allocator=a)
    c.mem_pool_host = NS(device_ptrs=torch.tensor([0], dtype=torch.uint64))
    data = torch.full((68, 4), -1, dtype=torch.int64)
    data[10], data[11] = 0, 1
    native = c._run_swap_in_kernel

    def refill(*args, **kwargs):
        result = native(*args, **kwargs)
        for token, loc in zip(
            c.req_device_buffer_tokens.flatten(),
            c.req_device_buffer_token_locs.flatten(),
        ):
            if token >= 0:
                data[loc] = token
        return result

    def transfer(src_ptrs, dst_ptrs, src, dst):
        data[dst] = data[src].clone()

    c._run_swap_in_kernel = refill
    monkeypatch.setattr(workspace, "transfer_cache_dsv4_mla", transfer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    saved_free, saved_release = a.free_pages, a.release_pages
    # The fifth unique item evicts row 0's token 0 in this tiny cache.
    raw = torch.tensor([[0, 2], [2, 3], [1, 4]], dtype=torch.int32)
    with workspace.ShadowKVWorkspace(c, width=3) as copies:
        obs = ForwardObservation(c, 2, "exact", 10, 0, copies)
        with obs.capture():
            result = c.swap_in_selected_pages_spec(
                req_pool_indices=torch.tensor([0]),
                compressed_seq_lens=torch.full((3,), 10),
                top_k_result=raw,
                layer_id=0,
                verify_lens_cpu=[3],
                output_buffer=None,
            )
        assert (
            data[result.long()].tolist() == raw.unsqueeze(-1).expand(-1, -1, 4).tolist()
        )
        assert data[10].tolist() == [4] * 4  # The original KV WAS evicted.
        assert obs.layers[0]["resident_after"] == [1, 2, 3, 4]
        assert obs.layers[0]["workspace_d2d_bytes"] == 6 * 584
        assert obs.layers[0]["miss_events"] == [
            {"row": 0, "ids": [2]},
            {"row": 1, "ids": [3]},
            {"row": 2, "ids": [4]},
        ]
    assert a.free_pages is saved_free and a.release_pages is saved_release


@pytest.mark.parametrize("failure", ["forward", "oom"])
def test_workspace_exception_restores_exact_free_list(monkeypatch, failure):
    a = allocator(size=8)
    saved_free, saved_release = a.free_pages, a.release_pages
    c = NS(top_k=2, token_to_kv_pool_allocator=NS(hisparse_attn_allocator=a))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with pytest.raises(RuntimeError), workspace.ShadowKVWorkspace(
        c, width=12 if failure == "oom" else 2
    ):
        raise RuntimeError("forward failed")
    assert a.free_pages is saved_free and a.release_pages is saved_release


def test_workspace_does_not_copy_current_window_or_missing_indices(monkeypatch):
    copied = []
    monkeypatch.setattr(
        workspace, "transfer_cache_dsv4_mla", lambda *args: copied.append(args)
    )
    a = allocator()
    c = NS(
        top_k=4,
        token_to_kv_pool_allocator=NS(hisparse_attn_allocator=a),
        mem_pool_host=NS(device_ptrs=torch.tensor([0], dtype=torch.uint64)),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    with workspace.ShadowKVWorkspace(c, 1) as copies:
        raw = torch.tensor([[3, 10, -1, 4]])
        selected = torch.tensor([[30, 99, -1, -1]], dtype=torch.int32)
        tokens, locations = copies.copy_row(0, 0, raw, selected, history_units=10)
        assert tokens == [3] and locations == [copies.rows[0, 0].item()]
        assert selected.tolist() == [[locations[0], 99, -1, -1]]
        assert copied[0][2].tolist() == [30]


def test_widening_is_opt_in_and_does_not_lower_native_capacity(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_OASISKV_TRACE_CONFIG", raising=False)
    assert workspace.trace_resident_verify_width(12) == 12
    path = tmp_path / "config.json"
    monkeypatch.setenv("SGLANG_OASISKV_TRACE_CONFIG", str(path))
    path.write_text(json.dumps({"verify_candidates": 5}))
    assert workspace.trace_resident_verify_width(12) == 12
    path.write_text(json.dumps({"verify_candidates": 5, "probe_workspace": "row-copy"}))
    assert workspace.trace_resident_verify_width(12) == 6
    path.write_text(
        json.dumps({"verify_candidates": 12, "probe_workspace": "row-copy"})
    )
    with pytest.raises(ValueError, match="fit the runtime"):
        workspace.trace_resident_verify_width(12)


def test_plan_w6_p6_extends_proposal_but_not_real_residency():
    args = NS(
        mode="paired",
        verify_width=6,
        probes=list(range(1, 7)),
        input_lengths=[16384, 32768],
        context_length=40960,
        output_length=256,
        requests=1,
        snapshot_gib=32,
        max_windows=8,
        resident_units=3584,
        top_k=512,
        mem_fraction=0.9,
        prefetch_units="all",
        gpus="0,1,2,3",
    )
    assert runner.validate_args(args, 5) == 6
    assert runner.proposal_gamma(args, 5) == 12
    assert runner.resident_verify_width(args, 5) == 7
    args.resident_units = 2560
    with pytest.raises(ValueError, match="Real verify requires capacity"):
        runner.validate_args(args, 5)
    args.mode = "full-control"
    assert runner.proposal_gamma(args, 5) == 5
    assert runner.resident_verify_width(args, 5) == 6


def test_runtime_proposer_generates_twelve_candidates_from_five_token_config(
    monkeypatch,
):
    from sglang.srt.models.dspark import VanillaMarkov
    from sglang.srt.speculative.dspark_components import dspark_draft
    from sglang.srt.speculative.dspark_components.dspark_config import (
        resolve_runtime_config,
    )

    config = NS(dspark_block_size=5, dspark_markov_rank=2, dspark_noise_token_id=15)
    runtime = resolve_runtime_config(
        draft_hf_config=config, speculative_num_draft_tokens=13, target_vocab_size=16
    )
    assert runtime.gamma == 12 and config.dspark_block_size == 5
    head = VanillaMarkov(vocab_size=16, markov_rank=2)
    with torch.no_grad():
        head.markov_w2.weight.zero_()
    seen = []

    def forward(batch):
        seen.append(batch)
        return NS(
            logits_output=NS(hidden_states=batch.input_embeds), can_run_graph=False
        )

    def logits(hidden):
        assert hidden.shape == (12, 4)
        result = torch.full((12, 16), -100.0)
        result[torch.arange(12), torch.arange(1, 13)] = 100
        return result, None

    monkeypatch.setattr(dspark_draft, "_make_num_token_non_padded", lambda *args: None)
    proposer = dspark_draft.DraftBlockProposer(
        draft_model=NS(gamma=5, markov_head=head, compute_base_logits=logits),
        draft_model_runner=NS(forward=forward, decode_cuda_graph_runner=None),
        gamma=runtime.gamma,
        mask_token_id=runtime.mask_token_id,
        draft_block_spec_info=None,
    )
    result = proposer.propose(
        batch=NS(
            seq_lens=torch.tensor([16384]),
            seq_lens_cpu=torch.tensor([16384]),
            req_pool_indices=torch.tensor([0]),
            can_run_dp_cuda_graph=False,
        ),
        draft_input=NS(bonus_tokens=torch.tensor([7])),
        verify_window=NS(
            positions_2d=torch.arange(16384, 16397).view(1, 13),
            verify_cache_loc_2d=torch.arange(13).view(1, 13),
        ),
        bs=1,
        device="cpu",
        target_model=NS(get_input_embeddings=lambda: torch.nn.Embedding(16, 4)),
        sampling_info=None,
    )
    assert len(seen) == 1  # One wide draft forward, never W then P.
    assert seen[0].input_ids.tolist() == [7] + [15] * 11
    assert seen[0].positions.tolist() == list(range(16384, 16396))
    assert result.draft_block.draft_tokens.tolist() == [list(range(1, 13))]


def test_workspace_bytes_are_separate_from_misses_and_protocols_cannot_mix():
    protocol = {
        "checkpoint_gamma": 5,
        "proposal_gamma": 11,
        "token_counting": "including-anchor",
        "probe_workspace": "row-copy",
    }
    source, following, probe = [dict(r, **protocol) for r in records()]
    probe.update(mode="exact", probe_workspace_d2d_bytes=12 * 512 * 584)
    before = analyzer.compare(
        source, following, dict(probe, probe_workspace_d2d_bytes=0)
    )
    after = analyzer.compare(source, following, probe)
    assert after.pop("probe_workspace_d2d_bytes") == 12 * 512 * 584
    before.pop("probe_workspace_d2d_bytes")
    assert after == before
    report = analyzer.analyze([source, following, probe])
    assert report["proposal_protocol"] == protocol
    with pytest.raises(ValueError, match="Cannot mix proposal widths"):
        analyzer.analyze([source, following, dict(probe, proposal_gamma=5)])
    with pytest.raises(ValueError, match="must not use the exact shadow workspace"):
        analyzer.validate(dict(probe, mode="resident", probe_demand_h2d_bytes=0))


def test_audit_rejects_inconsistent_runtime_proposal_geometry(tmp_path):
    directory = tmp_path / "probes"
    write_paired_run(directory)
    path = directory / "runtime.json"
    runtime = json.loads(path.read_text())
    path.write_text(json.dumps(dict(runtime, proposal_gamma=11)))
    with pytest.raises(ValueError, match="Runtime proposal/workspace policy differs"):
        auditor.audit_within_server(directory)
