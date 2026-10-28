"""Persistent request descriptors and small per-replay OasisKV input uploads."""

from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels


@dataclass(eq=False)
class GraphRequest:
    owner: object
    slot: int
    generation: int
    index: int
    states: tuple = ()
    length: int = 0
    step_delta: int = 0
    limit: int = 0
    dirty: bool = True
    released: bool = False

    def invalidate(self, *, release=False):
        self.dirty = True
        if release and not self.released:
            self.released = True
            self.owner.forget(self)


class GraphInputs:
    def __init__(self, graph):
        self.graph = graph
        self.requests = {}
        self.capacity = graph.cache.max_requests or graph.max_bs
        self.free = list(range(self.capacity - 1, -1, -1))
        self.num_layers = graph.tables.shape[0]
        self.descriptors = torch.zeros(
            (self.capacity, self.num_layers, kernels.FIELDS + 1),
            dtype=torch.int64,
            device=graph.device,
        )
        self.descriptors[..., -1] = -1
        self.batch = torch.full(
            (graph.max_bs, 4), -1, dtype=torch.int64, device=graph.device
        )
        # Only the graph prologue reads these inputs. Protect the next upload
        # from that reader, not from the much longer model or late prefetch IO.
        self.consumed = torch.cuda.Event(external=True) if graph.cuda else None
        if self.consumed is not None:
            self.consumed.record()
        self.uploads = []
        for _ in range(2):
            descriptors = torch.empty(
                self.descriptors.shape, dtype=torch.int64, pin_memory=graph.cuda
            )
            batch = torch.empty(
                self.batch.shape, dtype=torch.int64, pin_memory=graph.cuda
            )
            self.uploads.append(
                (
                    descriptors,
                    descriptors.numpy(),
                    batch,
                    batch.numpy(),
                    torch.cuda.Event() if graph.cuda else None,
                )
            )
        self.upload_index = 0

    def forget(self, request):
        if self.requests.get(request.slot) is request:
            del self.requests[request.slot]
            self.free.append(request.index)
        request.states = ()

    def _refresh(self, request, prefix, host, view, waits):
        graph = self.graph
        states, rows = [], [[0] * kernels.FIELDS + [-1] for _ in range(self.num_layers)]
        limit = graph.max_context
        for layer_id, layout in graph.layers.items():
            layer = layer_id - graph.start_layer
            identity = (request.slot, request.generation, layer_id)
            state = graph.cache.layers.get(identity)
            if state is None:
                if prefix:
                    raise ValueError(f"Missing OasisKV prefill history for {identity}")
                state = graph.cache.new_layer(
                    layer_id,
                    layout.heads,
                    layout.key_dim,
                    layout.value_dim,
                    layout.dtype,
                    graph.device,
                )
                graph.cache.layers[identity] = state
            if state.length != prefix:
                raise ValueError(
                    f"OasisKV cache length {state.length} != prefix {prefix}"
                )
            if (state.heads, state.key_dim, state.value_dim, state.dtype) != (
                layout.heads,
                layout.key_dim,
                layout.value_dim,
                layout.dtype,
            ):
                raise ValueError(
                    "OasisKV prefill storage does not match captured KV geometry"
                )
            # Only registration, growth, dense page movement or an eager handoff
            # reaches these fences. Ordinary graph steps use captured done waits.
            waits[layer_id].extend(
                event
                for event in (
                    state.selection_done,
                    state.pending,
                    state.backup_done,
                    state.refill_done,
                    state.read_done,
                )
                if event is not None
            )
            state.graph_background = layout.background
            state.prepare_append(prefix + 1)
            rows[layer] = state.graph_row(prefix) + [request.generation]
            limit = min(limit, state.capacity * graph.config.block_size)
            if not state.sparse:
                limit = min(
                    limit,
                    graph.config.token_budget,
                    (state.page_table.active + 1) * graph.config.block_size,
                )
            states.append(state)
            state._graph_bindings.add(request)
        view[request.index] = rows
        self.descriptors[request.index].copy_(
            host[request.index], non_blocking=graph.cuda
        )
        request.states = tuple(states)
        request.length, request.step_delta, request.limit = prefix, 0, limit
        request.dirty = False

    def prepare(self, slots, generations, prefixes, paired, graph_batch_size):
        graph = self.graph
        if not len(slots) == len(generations) == len(prefixes):
            raise ValueError("OasisKV graph request metadata lengths disagree")
        if (
            len(slots) > graph_batch_size
            or graph_batch_size > graph.max_bs
            or len(set(slots)) != len(slots)
        ):
            raise ValueError(
                "OasisKV graph requires distinct live slots within max batch"
            )
        host, view, batch_host, batch_view, uploaded = self.uploads[self.upload_index]
        self.upload_index = 1 - self.upload_index
        if uploaded is not None and not uploaded.query():
            uploaded.synchronize()
        if self.consumed is not None:
            torch.cuda.current_stream(graph.device).wait_event(self.consumed)
        batch_view.fill(-1)
        pending, refreshes = [], []
        for slot, generation, prefix in zip(slots, generations, prefixes):
            if prefix < 0 or prefix >= graph.max_context:
                raise ValueError(
                    "OasisKV decode position exceeds captured context capacity"
                )
            request = self.requests.get(slot)
            if request is not None and request.generation != generation:
                request.invalidate(release=True)
                request = None
            if request is None:
                if not self.free:
                    raise RuntimeError("OasisKV graph request registry exhausted")
                request = GraphRequest(self, slot, generation, self.free.pop())
                self.requests[slot] = request
            if request.dirty or prefix + 1 > request.limit:
                refreshes.append((request, prefix))
            elif request.length != prefix:
                raise ValueError(
                    f"OasisKV cache length {request.length} != prefix {prefix}"
                )
            pending.append(request)
        if refreshes:
            waits = {layer_id: [] for layer_id in graph.layers}
            for request, prefix in refreshes:
                self._refresh(request, prefix, host, view, waits)
            # Preserve layer-local readiness: a late eager writer must not hold
            # earlier layers behind a model-wide barrier.
            if graph.cuda:
                stream = graph.prepare_stream or torch.cuda.current_stream(graph.device)
                for layer_id, events in waits.items():
                    for event in dict.fromkeys(events):
                        stream.wait_event(event)
                    background = graph.layers[layer_id].background
                    if background is not None:
                        background.inputs_ready.record(stream)
        for b, (request, prefix) in enumerate(zip(pending, prefixes)):
            batch_view[b] = (
                request.index,
                request.generation,
                prefix,
                request.step_delta,
            )
        self.batch.copy_(batch_host, non_blocking=graph.cuda)
        if uploaded is not None:
            uploaded.record(torch.cuda.current_stream(graph.device))
        graph.replay_batch_size, graph.replay_paired = graph_batch_size, paired
        graph.pending = pending
        if not graph.cuda:
            self.materialize(graph_batch_size)

    def materialize(self, batch_size):
        graph = self.graph
        if graph.device.type == "cuda":
            kernels.prepare_request_tables[(self.num_layers, (batch_size + 15) // 16)](
                self.descriptors,
                self.batch,
                graph.tables,
                REQUESTS=self.capacity,
                LAYERS=self.num_layers,
                MAX_BATCH=graph.max_bs,
                BATCH=batch_size,
            )
            self.consumed.record()
        else:
            # Reference path for CPU tests. No host table reconstruction in
            # production: the small gather is captured once in the target graph.
            index, generation, prefix, delta = self.batch[:batch_size].unbind(1)
            rows = self.descriptors[index.clamp(0, self.capacity - 1)].permute(1, 0, 2)
            valid = (index >= 0) & (index < self.capacity) & (prefix >= 0)
            valid = valid[None] & (rows[..., -1] == generation[None])
            target = graph.tables[:, :batch_size]
            target.copy_(torch.where(valid[..., None], rows[..., :-1], 0))
            target[..., kernels.PREFIX] = torch.where(valid, prefix[None], -1)
            target[..., kernels.STEP] += torch.where(valid, delta[None], 0)
