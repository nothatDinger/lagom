"""Capture V4's per-layer selection and direct host-to-resident transfers.

The existing HiSparse kernels own real miss refill, read-only LRU planning and
the V4 paged KV layout. This class supplies fixed scratch and stream edges; it
does not introduce a second KV buffer. Graph and eager execution share the
coordinator's ring scratch, and are fenced outside capture when switching modes.
"""

import torch
import triton
from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla
from sglang.srt.layers.attention.triton_ops.oasiskv_v4 import (
    filter_history_candidates,
    publish_hot_metadata,
)


class OasisKVV4GraphState:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.select_stream = coordinator._oasiskv_select_stream
        self.transfer_stream = coordinator._previous_prefetch_stream
        self.done = torch.cuda.Event()
        self.pending = False
        self.req_slots = set()
        self.waited_streams = set()
        self.eager_tasks = []

    def prepare_replay(self):
        """Fence eager work even if an earlier stream already consumed its task.

        Consuming enqueues a wait; it does not complete DMA on the CPU. The graph
        may run on a different stream and reuse ring slot zero, so it must join
        every outstanding ring event before using either the scratch or live KV.
        """
        self.wait()
        main = torch.cuda.current_stream()
        self.eager_tasks = []
        for slots in self.coordinator._oasiskv_ring:
            for task in slots:
                if task.pending_event is not None:
                    task.pending_event.wait(main)
                    self.eager_tasks.append(task)

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
        # All arguments are graph-owned tensors or persistent coordinator buffers.
        # Do not bind CPU identities, rotate a task ring or count work at capture.
        self.select_stream.wait_stream(torch.cuda.current_stream())
        predicted_c4_entries.record_stream(self.select_stream)
        compressed_seq_lens.record_stream(self.select_stream)
        with torch.cuda.stream(self.select_stream):
            filter_history_candidates[(bs,)](
                predicted_c4_entries,
                forward_batch.seq_lens,
                forward_batch.oasiskv_draft_valid,
                c.num_real_reqs,
                task.predicted_entries,
                predicted_c4_entries.stride(0),
                k,
                c.compress_ratio,
                triton.next_power_of_2(k),
            )
            c._run_swap_in_kernel(
                forward_batch.req_pool_indices,
                compressed_seq_lens,
                task.predicted_entries[:bs],
                layer_id,
                record_plan=True,
                num_top_k=k,
                output_buffer=task.device_locs,
                miss_plan=(task.miss_src, task.miss_dst, task.miss_count),
                skip_io=True,
                planned_metadata=(task.planned_tokens[:bs], task.planned_lru[:bs]),
                num_real_reqs=c.num_real_reqs,
            )
            task.selection_event.record(self.select_stream)

    def transfer(self, forward_batch, *, layer_id):
        """Fork after this layer's attention; publish only after copying misses."""
        c = self.coordinator
        task = c._oasiskv_ring[layer_id][0]
        bs = forward_batch.req_pool_indices.numel()
        task.selection_event.wait(self.transfer_stream)
        self.transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.transfer_stream):
            copy_cache_planned_mla(
                miss_src=task.miss_src[:bs],
                miss_dst=task.miss_dst[:bs],
                miss_count=task.miss_count[:bs],
                num_real_reqs=c.num_real_reqs,
                host_cache=c.mem_pool_host.kv_buffer[layer_id],
                device_buffer=c.mem_pool_device.kv_buffer[layer_id],
                item_size_bytes=c.item_size_bytes,
                num_blocks=4,
                is_dsv4_layout=True,
                skip_io=c.skip_io,
            )
            publish_hot_metadata[(bs,)](
                forward_batch.req_pool_indices,
                c.num_real_reqs,
                task.planned_tokens,
                task.planned_lru,
                c.req_device_buffer_tokens[layer_id],
                c.lru_slots[layer_id],
                c.device_buffer_size,
                c.req_device_buffer_tokens.stride(1),
                triton.next_power_of_2(c.device_buffer_size),
            )

    def finish_in_graph(self):
        """Join both capture branches before returning to root-only C4 commit.

        For short histories selection copies unchanged hot metadata and does no
        I/O. For long histories it updates only [0, hot_size), while C4 promotion
        uses hot_size, the reserved newest slot. In both cases commit runs after
        this join, so it cannot be overwritten by an old metadata snapshot.
        """
        main = torch.cuda.current_stream()
        main.wait_stream(self.select_stream)
        main.wait_stream(self.transfer_stream)

    def finish_replay(self, forward_batch):
        self.done.record(torch.cuda.current_stream())
        self.pending = True
        self.waited_streams.clear()
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

    def wait(self):
        stream = torch.cuda.current_stream()
        if self.pending and stream not in self.waited_streams:
            self.done.wait(stream)
            self.waited_streams.add(stream)

    def drain_request(self, req_slot):
        if self.pending and req_slot in self.req_slots:
            # Host-page reclamation is immediate CPU work, unlike stream waits.
            self.done.synchronize()
            self.pending = False
            self.req_slots.clear()
