"""Independent transfers released by each layer of a Lagom target graph.

The target records external readiness events after producing disjoint demand
and prefetch copy plans. A separately launched graph waits for those events on
the transfer stream. It can therefore copy future KV while the target is still
refilling and evaluating its current window, without a foreground graph join.
"""

from dataclasses import dataclass
from typing import Callable

import torch

from sglang.srt.observability.lagom_perf import phase, register_side_stream


@dataclass
class _Layer:
    ready: torch.cuda.Event
    done: torch.cuda.Event
    pending: bool = False


class LagomPrefetchGraphs:
    def __init__(self, transfer_stream):
        self.transfer_stream = transfer_stream
        register_side_stream(transfer_stream)
        self.layers: dict[int, _Layer] = {}
        self.functions: dict[int, dict[int, Callable[[], None]]] = {}
        self.graphs: dict[int, tuple[torch.cuda.CUDAGraph, tuple[int, ...]]] = {}
        self._failure: Exception | None = None

    @property
    def pending(self) -> bool:
        return any(layer.pending for layer in self.layers.values())

    def fail(self, error: Exception) -> None:
        # The planner publishes residency before the transfers run. A failed
        # copy invalidates that promise, so cache reuse requires reconstruction.
        if self._failure is None:
            self._failure = error

    def check_healthy(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "Lagom cache transfer failed; rebuild the coordinator before reuse"
            ) from self._failure

    def _layer(self, layer_id: int) -> _Layer:
        layer = self.layers.get(layer_id)
        if layer is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Lagom layer events must be initialized during warmup"
                )
            layer = _Layer(
                torch.cuda.Event(external=True), torch.cuda.Event(external=True)
            )
            layer.ready.record()
            layer.done.record()
            self.layers[layer_id] = layer
        return layer

    def consume_layer(self, layer_id: int) -> None:
        self.check_healthy()
        # Unconditional during capture: the next target must not overwrite a
        # prior transfer's plan or read a resident slot whose H2D is unfinished.
        with phase("prefetch_wait", layer=layer_id):
            self._layer(layer_id).done.wait(torch.cuda.current_stream())

    def plan_ready(
        self,
        layer_id: int,
        batch_size: int,
        transfer: Callable[[], None],
        *,
        capturing: bool,
    ) -> None:
        self.check_healthy()
        layer = self._layer(layer_id)
        self.functions.setdefault(batch_size, {})[layer_id] = transfer
        layer.ready.record(torch.cuda.current_stream())
        if not capturing:
            # A launch can fail after earlier copies were enqueued. Retain a
            # drainable completion fence even on the exception path.
            layer.pending = True
            with torch.cuda.stream(self.transfer_stream):
                layer.ready.wait(self.transfer_stream)
                try:
                    transfer()
                except Exception as exc:
                    self.fail(exc)
                    raise
                finally:
                    layer.done.record(self.transfer_stream)

    def has_shape(self, batch_size: int) -> bool:
        return batch_size in self.graphs

    def capture(self, batch_size: int) -> None:
        """Capture after target capture, with stable coordinator-owned plans."""
        self.check_healthy()
        if self.has_shape(batch_size):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Lagom side graph capture must follow target capture")
        functions = self.functions.get(batch_size)
        if not functions:
            raise RuntimeError("Lagom target did not register any CSA transfer plans")
        self.synchronize()
        # Warm transfer kernels outside capture. Startup request counts are
        # inactive, and the stream edge makes initialized plan storage visible.
        self.transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.transfer_stream):
            for transfer in functions.values():
                transfer()
        self.transfer_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.transfer_stream):
            for layer_id, transfer in functions.items():
                layer = self.layers[layer_id]
                layer.ready.wait(self.transfer_stream)
                transfer()
                layer.done.record(self.transfer_stream)
        self.graphs[batch_size] = (graph, tuple(functions))

    def replay(self, batch_size: int) -> None:
        """Enqueue immediately after target launch; never wait for its tail."""
        self.check_healthy()
        graph, layer_ids = self.graphs[batch_size]
        for layer_id in layer_ids:
            self.layers[layer_id].pending = True
        with torch.cuda.stream(self.transfer_stream):
            try:
                graph.replay()
            except Exception as exc:
                self.fail(exc)
                # Fence any work already enqueued by a failed launch before
                # request cleanup is allowed to reclaim its host source pages.
                for layer_id in layer_ids:
                    self.layers[layer_id].done.record(self.transfer_stream)
                raise

    def wait(self) -> None:
        """Order cache mutation after transfers without blocking the CPU."""
        for layer in self.layers.values():
            layer.done.wait(torch.cuda.current_stream())

    def synchronize(self) -> None:
        """Drain actual DMA completion before request/host-page reclamation."""
        for layer in self.layers.values():
            if layer.pending:
                layer.done.synchronize()
                layer.pending = False
