"""Graph/event topology without requiring a CUDA runtime."""

from contextlib import contextmanager
from unittest.mock import MagicMock

import torch
from sglang.srt.mem_cache.oasiskv_async import LayerPrefetchGraphs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_captures_independent_selection_transfer_graphs_and_external_waits(monkeypatch):
    current, calls = ["main"], []

    @contextmanager
    def stream_context(stream):
        old, current[0] = current[0], stream.name
        yield
        current[0] = old

    @contextmanager
    def graph_context(graph, stream):
        with stream_context(stream):
            calls.append(("capture", current[0]))
            yield

    events = []

    def event_factory(*, external):
        assert external
        event = MagicMock()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", event_factory)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: MagicMock())
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "main")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    select, transfer = MagicMock(), MagicMock()
    select.name, transfer.name = "select", "transfer"
    pipe = LayerPrefetchGraphs(select, transfer)
    pipe.capture(
        4,
        lambda: calls.append(("select", current[0])),
        lambda: calls.append(("copy", current[0])),
    )
    assert calls == [
        ("select", "select"),
        ("copy", "transfer"),
        ("capture", "select"),
        ("select", "select"),
        ("capture", "transfer"),
        ("copy", "transfer"),
    ]
    pipe.prediction_ready.wait.assert_called_once_with()
    pipe.selection_done.wait.assert_called_once_with()
    pipe.read_done.wait.assert_called_once_with()
    pipe.consume()
    pipe.inputs_ready.wait.assert_called_once_with()
    pipe.done.wait.assert_called_once_with()
    pipe.replay(4)
    assert pipe.pending
    for graph in pipe.graphs[4]:
        graph.replay.assert_called_once_with()
    before = len(calls)
    pipe.capture(4, None, None)
    assert len(calls) == before  # no recapture when changing request contents
    pipe.synchronize()
    pipe.done.synchronize.assert_called_once_with()
    select.wait_stream.assert_called_once_with("main")  # startup allocations only
    transfer.wait_stream.assert_not_called()
