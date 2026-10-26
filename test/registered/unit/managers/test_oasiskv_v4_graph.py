"""V4 graph stream dependencies and host-page lifetime, without a GPU."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.oasiskv_v4_graph import OasisKVV4GraphState
from sglang.srt.speculative.oasiskv_lookahead import (
    plan_oasiskv_layer_prefetch,
    submit_oasiskv_layer_prefetch,
    submit_oasiskv_pending_prefetches,
)


def test_graph_layer_hooks_submit_once_even_at_c4_boundary():
    state = SimpleNamespace(plan=MagicMock(), transfer=MagicMock())
    c = SimpleNamespace(oasiskv_graph_state=state, _active_dspark_window=object())
    kwargs = {
        "layer_id": 1,
        "predicted_c4_entries": object(),
        "compressed_seq_lens": object(),
    }
    fb = SimpleNamespace(
        is_oasiskv_graph_capture=True, _oasiskv_pending_prefetch={7: (c, kwargs)}
    )
    assert plan_oasiskv_layer_prefetch(fb, 7)
    assert plan_oasiskv_layer_prefetch(fb, 7)
    state.plan.assert_called_once_with(
        fb,
        layer_id=1,
        predicted_c4_entries=kwargs["predicted_c4_entries"],
        compressed_seq_lens=kwargs["compressed_seq_lens"],
    )
    assert submit_oasiskv_layer_prefetch(fb, 7)
    state.transfer.assert_called_once_with(fb, layer_id=1)
    assert not submit_oasiskv_layer_prefetch(fb, 7)
    assert fb._oasiskv_pending_prefetch == {}
    # Root-only commit must not enqueue the captured transfer a second time.
    submit_oasiskv_pending_prefetches(fb)


def test_graph_snapshots_prediction_inputs_and_launches_without_tail_join(monkeypatch):
    import sglang.srt.managers.oasiskv_v4_graph as module

    calls = []

    class Graphs:
        def __init__(self, *args):
            self.prediction_ready = MagicMock()
            self.read_done = MagicMock()
            self.selection_done = MagicMock()
            self.done = MagicMock()
            self.consume = MagicMock()
            self.synchronize = MagicMock()
            self.graphs = {}

        def capture(self, shape, select, transfer):
            self.graphs.setdefault(shape, (select, transfer))

        def replay(self, shape):
            select, transfer = self.graphs[shape]
            select()
            transfer()

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kw):
                calls.append(self.name)
                if self.name == "publish":
                    assert args[1] is state.layers[0].real_count
                    assert kw["SKIP_SHORT"]
                    assert kw["lengths"] is state.layers[0].lengths

            return launch

    monkeypatch.setattr(torch.cuda, "Event", MagicMock())
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    monkeypatch.setattr(module, "LayerPrefetchGraphs", Graphs)
    monkeypatch.setattr(module, "filter_history_candidates", Kernel("filter"))
    monkeypatch.setattr(module, "publish_hot_metadata", Kernel("publish"))

    def copy(**kw):
        assert kw["num_real_reqs"] is state.layers[0].real_count
        assert kw["is_dsv4_layout"]
        calls.append("copy")

    def plan(reqs, lengths, *args, **kw):
        assert kw["skip_io"] and kw["record_plan"]
        assert kw["num_real_reqs"] is state.layers[0].real_count
        assert reqs.tolist() == [1, 1]
        assert lengths.tolist() == [8, 0]
        calls.append("plan")

    monkeypatch.setattr(module, "copy_cache_planned_mla", copy)
    task = SimpleNamespace(
        predicted_entries=torch.empty(3, 2, dtype=torch.int32),
        planned_tokens=torch.empty(3, 4, dtype=torch.int32),
        planned_lru=torch.empty(3, 4, dtype=torch.int16),
        device_locs=torch.empty(3, 2, dtype=torch.int32),
        miss_src=torch.empty(3, 2, dtype=torch.int64),
        miss_dst=torch.empty(3, 2, dtype=torch.int32),
        miss_count=torch.empty(3, dtype=torch.int32),
    )
    c = SimpleNamespace(
        _oasiskv_ring=[[task]],
        _oasiskv_select_stream=object(),
        _previous_prefetch_stream=object(),
        num_real_reqs=torch.tensor([1]),
        compress_ratio=4,
        prefetcher=SimpleNamespace(logical_entries=2),
        device_buffer_size=4,
        req_device_buffer_tokens=torch.empty(1, 3, 5),
        lru_slots=torch.empty(1, 3, 4),
        mem_pool_host=SimpleNamespace(kv_buffer=[object()]),
        mem_pool_device=SimpleNamespace(kv_buffer=[object()]),
        item_size_bytes=584,
        skip_io=False,
        _run_swap_in_kernel=plan,
    )
    state = OasisKVV4GraphState(c)
    fb = SimpleNamespace(
        req_pool_indices=torch.tensor([1, 1]),
        req_pool_indices_cpu=torch.tensor([1]),
        seq_lens=torch.tensor([31, 0]),
        oasiskv_draft_valid=torch.tensor([True, True]),
    )
    state.plan(
        fb,
        layer_id=0,
        predicted_c4_entries=torch.zeros(2, 2),
        compressed_seq_lens=torch.tensor([8, 0]),
    )
    side = state.layers[0].graphs
    side.prediction_ready.record.assert_called_once_with()
    state.transfer(fb, layer_id=0)
    side.read_done.record.assert_called_once_with()
    state.finish_in_graph()
    assert calls == []  # no selection/transfer launch or model-tail join
    side.done.wait.assert_not_called()
    # Shared runner inputs may be prepared for another batch after launch.
    fb.req_pool_indices.fill_(2)
    c.num_real_reqs.fill_(0)
    state.finish_replay(fb)
    assert calls == ["filter", "plan", "copy", "publish"]
    assert state.layers[0].real_count.item() == 1
    state.consume_layer(0)
    side.consume.assert_called_once_with()
    state.wait_selection_for_commit()
    side.selection_done.wait.assert_called_once_with()
    side.done.wait.assert_not_called()


def test_graph_completion_protects_all_prior_requests_until_reclaimed(monkeypatch):
    event = MagicMock()
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    c = SimpleNamespace(_oasiskv_select_stream=None, _previous_prefetch_stream=None)
    state = OasisKVV4GraphState(c)
    side = MagicMock()
    state.layers[0] = SimpleNamespace(graphs=side)

    def batch(slots):
        return SimpleNamespace(
            req_pool_indices_cpu=torch.tensor(slots),
            req_pool_indices=torch.tensor(slots),
            batch_size=len(slots),
        )

    state.finish_replay(batch([3, 7]))
    state.wait(0)
    side.done.wait.assert_called_once_with()
    state.finish_replay(batch([7]))
    state.replay_batch_size = 1
    state.finish_replay(batch([]))
    state.drain_request(9)
    event.synchronize.assert_not_called()
    state.drain_request(3)
    event.synchronize.assert_called_once()
    side.synchronize.assert_called_once()
    assert not state.pending and not state.req_slots


def test_coordinator_drains_graph_before_host_reclamation():
    c = HiSparseCoordinator.__new__(HiSparseCoordinator)
    c.oasiskv_graph_state = MagicMock()
    c._has_pending_dspark_commit = True
    c._dspark_commit_done_event = MagicMock()
    c._drain_oasiskv_tasks_for_request(7)
    c.oasiskv_graph_state.drain_request.assert_called_once_with(7)
    c._dspark_commit_done_event.synchronize.assert_called_once()
    assert not c._has_pending_dspark_commit


def test_graph_joins_consumed_eager_task_and_retains_its_request_owner(monkeypatch):
    event, done, main = MagicMock(), MagicMock(), object()
    monkeypatch.setattr(torch.cuda, "Event", lambda: done)
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    # Consumed by a different stream, but its asynchronous transfer may still run.
    task = SimpleNamespace(
        valid=False, planning=False, pending_event=event, req_slots=torch.tensor([9])
    )
    c = SimpleNamespace(
        _oasiskv_select_stream=None,
        _previous_prefetch_stream=None,
        _oasiskv_ring=[[task]],
    )
    state = OasisKVV4GraphState(c)
    state.prepare_replay()
    event.wait.assert_called_once_with(state.prepare_stream)
    assert task.pending_event is event
    state.finish_replay(
        SimpleNamespace(
            req_pool_indices_cpu=torch.tensor([3]), req_pool_indices=torch.tensor([3])
        )
    )
    assert task.pending_event is None
    state.drain_request(9)
    done.synchronize.assert_called_once()


def test_idle_target_replay_still_fences_captured_task_scratch():
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
        DecodeCudaGraphRunner,
    )

    state = MagicMock()
    coordinator = SimpleNamespace(
        prefetcher_name="oasiskv",
        oasiskv_graph_state=state,
        mem_pool_device=SimpleNamespace(layer_num=1),
        consume_oasiskv_prefetch=MagicMock(),
    )
    fb = SimpleNamespace(
        forward_mode=ForwardMode.IDLE,
        batch_size=0,
        is_oasiskv_paired=False,
        req_pool_indices=torch.empty(0, dtype=torch.int64),
        req_pool_indices_cpu=None,
        seq_lens_cpu=None,
    )
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(hisparse_coordinator=coordinator)
    runner._prepare_oasiskv_graph_replay(fb)
    state.prepare_replay.assert_called_once()
    coordinator.consume_oasiskv_prefetch.assert_not_called()
    backend = object.__new__(DeepseekV4AttnBackend)
    backend.is_draft_runner, backend.hisparse_coordinator = False, coordinator
    backend.finish_cuda_graph_replay(fb)
    state.finish_replay.assert_called_once_with(fb)
