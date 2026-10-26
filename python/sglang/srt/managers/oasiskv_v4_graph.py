"""Capture V4's per-layer selection and direct host-to-resident transfers.

The existing HiSparse kernels own real miss refill, read-only LRU planning and
the V4 paged KV layout. This class supplies fixed scratch and stream edges; it
does not introduce a second KV buffer. Graph and eager execution share the
coordinator's ring scratch, and are fenced outside capture when switching modes.
"""

from types import SimpleNamespace

import torch
import triton
from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.srt.layers.attention.triton_ops.oasiskv_v4 import (
    filter_history_candidates,
    publish_hot_metadata,
)
from sglang.srt.mem_cache.oasiskv_async import LayerPrefetchGraphs


class OasisKVV4GraphState:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.select_stream = coordinator._oasiskv_select_stream
        self.transfer_stream = coordinator._previous_prefetch_stream
        self.done = torch.cuda.Event()
        self.pending = False
        self.req_slots = set()
        self.eager_tasks = []
        self.layers = {}
        self.prepare_stream = torch.cuda.Stream()
        self.replay_batch_size = None

    def prepare_replay(self):
        # Keep per-layer readiness fences off the foreground launch stream.
        self.eager_tasks = []
        for layer_id, slots in enumerate(self.coordinator._oasiskv_ring):
            for task in slots:
                if task.pending_event is not None:
                    task.pending_event.wait(self.prepare_stream)
                    self.eager_tasks.append(task)
            if layer_id in self.layers:
                self.layers[layer_id].graphs.inputs_ready.record(self.prepare_stream)

    def consume_layer(self, layer_id):
        if layer_id in self.layers:
            self.layers[layer_id].graphs.consume()

    def plan(
        self, forward_batch, *, layer_id, predicted_c4_entries, compressed_seq_lens
    ):
        """Fork after real refill; snapshot metadata without touching live KV."""
        c = self.coordinator
        task = c._oasiskv_ring[layer_id][0]
        bs = forward_batch.req_pool_indices.numel()
        k = c.prefetcher.logical_entries
        if predicted_c4_entries.shape[1] < k:
            raise ValueError(
                "V4 OasisKV prediction is narrower than the prefetch budget"
            )
        if layer_id not in self.layers:
            capacity = task.predicted_entries.shape[0]

            def vector(source):
                return torch.empty(capacity, dtype=source.dtype, device=source.device)

            self.layers[layer_id] = SimpleNamespace(
                graphs=LayerPrefetchGraphs(self.select_stream, self.transfer_stream),
                reqs=vector(forward_batch.req_pool_indices),
                prefixes=vector(forward_batch.seq_lens),
                valid=vector(forward_batch.oasiskv_draft_valid),
                lengths=vector(compressed_seq_lens),
                predicted=torch.empty_like(task.predicted_entries),
                real_count=torch.zeros_like(c.num_real_reqs),
            )
        layer = self.layers[layer_id]
        if bs not in layer.graphs.graphs:
            layer.graphs.synchronize()
            layer.real_count.zero_()

        def select():
            filter_history_candidates[(bs,)](
                layer.predicted,
                layer.prefixes,
                layer.valid,
                layer.real_count,
                task.predicted_entries,
                layer.predicted.stride(0),
                k,
                c.compress_ratio,
                triton.next_power_of_2(k),
            )
            c._run_swap_in_kernel(
                layer.reqs[:bs],
                layer.lengths[:bs],
                task.predicted_entries[:bs],
                layer_id,
                record_plan=True,
                num_top_k=k,
                output_buffer=task.device_locs,
                miss_plan=(task.miss_src, task.miss_dst, task.miss_count),
                skip_io=True,
                planned_metadata=(task.planned_tokens[:bs], task.planned_lru[:bs]),
                num_real_reqs=layer.real_count,
            )

        def transfer():
            copy_cache_planned_mla(
                miss_src=task.miss_src[:bs],
                miss_dst=task.miss_dst[:bs],
                miss_count=task.miss_count[:bs],
                num_real_reqs=layer.real_count,
                host_cache=c.mem_pool_host.kv_buffer[layer_id],
                device_buffer=c.mem_pool_device.kv_buffer[layer_id],
                item_size_bytes=c.item_size_bytes,
                num_blocks=4,
                is_dsv4_layout=True,
                skip_io=c.skip_io,
            )
            publish_hot_metadata[(bs,)](
                layer.reqs,
                layer.real_count,
                task.planned_tokens,
                task.planned_lru,
                c.req_device_buffer_tokens[layer_id],
                c.lru_slots[layer_id],
                c.device_buffer_size,
                c.req_device_buffer_tokens.stride(1),
                triton.next_power_of_2(c.device_buffer_size),
                lengths=layer.lengths,
                SKIP_SHORT=True,
            )

        layer.graphs.capture(bs, select, transfer)
        layer.reqs[:bs].copy_(forward_batch.req_pool_indices)
        layer.prefixes[:bs].copy_(forward_batch.seq_lens)
        layer.valid[:bs].copy_(forward_batch.oasiskv_draft_valid)
        layer.lengths[:bs].copy_(compressed_seq_lens)
        layer.predicted[:bs].copy_(predicted_c4_entries[:, :k])
        layer.real_count.copy_(c.num_real_reqs)
        layer.graphs.prediction_ready.record()

    def transfer(self, forward_batch, *, layer_id):
        self.layers[layer_id].graphs.read_done.record()

    def finish_in_graph(self):
        # No side stream participates in the foreground capture.
        pass

    def finish_replay(self, forward_batch):
        self.done.record(torch.cuda.current_stream())
        bs = self.replay_batch_size or forward_batch.req_pool_indices.numel()
        for layer in self.layers.values():
            layer.graphs.replay(bs)
        self.pending = True
        # A later graph waits on the previous completion before reusing scratch;
        # its event therefore also covers every older replay's host reads.
        if forward_batch.req_pool_indices_cpu is not None:
            self.req_slots.update(forward_batch.req_pool_indices_cpu.tolist())
        elif forward_batch.batch_size:
            raise RuntimeError(
                "V4 graph requires CPU request identities for reclamation"
            )
        for task in self.eager_tasks:
            # The graph's completion now covers these older request owners too.
            self.req_slots.update(task.req_slots.tolist())
            task.pending_event = None
        self.eager_tasks.clear()

    def wait(self, layer_id=None):
        layers = (
            self.layers.values()
            if layer_id is None
            else ([self.layers[layer_id]] if layer_id in self.layers else [])
        )
        for layer in layers:
            layer.graphs.done.wait()

    def wait_selection_for_commit(self):
        # A short-context plan snapshots fixed-slot metadata. Promotion may
        # update those slots once readers finish; transfer never publishes a
        # short-context snapshot and never writes the reserved newest slot.
        for layer in self.layers.values():
            layer.graphs.selection_done.wait()

    def drain_request(self, req_slot):
        if self.pending and req_slot in self.req_slots:
            self.done.synchronize()
            for layer in self.layers.values():
                layer.graphs.synchronize()
            self.pending = False
            self.req_slots.clear()
