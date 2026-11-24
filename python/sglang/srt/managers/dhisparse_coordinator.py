"""Adaptive DHiSparse serving state, shared arenas and residency lifecycle."""

from __future__ import annotations

from functools import wraps

import torch

from sglang.kernels.ops.kvcache import dhisparse as kernels
from sglang.srt.mem_cache.dhisparse import RequiredPageAllocator, ResidencyManager
from sglang.srt.observability.lagom_perf import SWAP_IN_COUNTERS, get_trace, phase
from sglang.srt.utils import get_device_module


def finish_dhisparse_attention(function):
    """Keep promotion inside graph capture and after the attention launch."""

    @wraps(function)
    def wrapped(self, q, k, v, layer, forward_batch, *args, **kwargs):
        result = function(self, q, k, v, layer, forward_batch, *args, **kwargs)
        coordinator = getattr(self, "hisparse_coordinator", None)
        adaptive = getattr(coordinator, "adaptive", None)
        if adaptive is not None:
            ratio, source, _ = self.token_to_kv_pool.layer_mapping[layer.layer_id]
            if ratio == 4:
                adaptive.finish_layer(source)
        return result

    return wrapped


class AdaptiveHiSparse:
    def __init__(self, coordinator, config):
        self.c, self.config = coordinator, config
        c = coordinator
        self.device_module = get_device_module()
        self.layout = c.mem_pool_device.dhisparse_layout
        if self.layout is None:
            raise RuntimeError(
                "Adaptive DHiSparse requires a budgeted shared-arena pool"
            )
        self.optional_locs = torch.full_like(c.req_to_host_pool, -1, dtype=torch.int32)
        n, layers, hot = (
            self.layout.num_requests,
            c.mem_pool_device.layer_num,
            c.device_buffer_size,
        )
        self.ages = torch.zeros(
            (layers, self.layout.slots, hot), dtype=torch.int64, device=c.device
        )
        self.epochs = torch.zeros(
            (layers, self.layout.slots), dtype=torch.int64, device=c.device
        )
        occurrences = self.layout.max_verify_width * self.layout.top_k
        self.trace = get_trace()
        self.workspaces = [
            kernels.allocate_workspace(
                n,
                occurrences,
                hot,
                c.device,
                track_hits=c.force_h2d_each_step or self.trace is not None,
            )
            for _ in range(self.layout.arenas)
        ]
        self.io_counts = [
            (
                torch.zeros(SWAP_IN_COUNTERS, dtype=torch.int32, device=c.device)
                if self.trace is not None
                else workspace["empty_counts"]
            )
            for workspace in self.workspaces
        ]
        self.outputs = [
            torch.full((n, occurrences), -1, dtype=torch.int32, device=c.device)
            for _ in range(self.layout.arenas)
        ]
        self.pending = {}
        self.prepared_requests = set()
        self.raw_allocator = c.token_to_kv_pool_allocator.hisparse_attn_allocator
        self.residency = ResidencyManager(
            page_size=c.page_size,
            config=config,
            available=self.raw_allocator.available_size,
            allocate=self.raw_allocator.alloc,
            copy=self._populate,
            publish=self._publish,
            unpublish=self._unpublish,
            backup=self._backup_history,
            retire=self._retire,
        )
        c.token_to_kv_pool_allocator.hisparse_attn_allocator = RequiredPageAllocator(
            self.raw_allocator, self.residency
        )

    def _publish(self, request, start, pages):
        self.optional_locs[request, start : start + len(pages)] = pages.to(torch.int32)

    def _unpublish(self, request):
        self.optional_locs[request].fill_(-1)

    def _retire(self, pages):
        # Page allocators run on the host. An event.wait alone would not stop
        # another stream from reusing a freed page before its readers finish.
        if not len(pages):
            return
        self.c.write_staging_stream.synchronize()
        if self.c.decode_producer_stream is not None:
            self.c.decode_producer_stream.synchronize()
        self.device_module.current_stream().synchronize()
        # Optional ownership is disjoint from the scheduler's batched frees.
        # Make reclaimed capacity available to the allocation that triggered it.
        previous = self.raw_allocator.is_not_in_free_group
        try:
            self.raw_allocator.is_not_in_free_group = True
            self.raw_allocator.free(pages)
        finally:
            self.raw_allocator.is_not_in_free_group = previous

    def _backup(self, host, device):
        c = self.c
        if not len(host):
            return
        from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla

        transfer_cache_dsv4_mla(
            src_ptrs=c.mem_pool_device.data_ptrs,
            dst_ptrs=c.mem_pool_host.data_ptrs,
            src_indices=device.to(torch.int64),
            dst_indices=host.to(torch.int64),
        )

    def _backup_history(self, state):
        self._backup(
            self.c.req_to_host_pool[state.request, : len(state.pages)], state.pages
        )
        self.device_module.current_stream().synchronize()

    def _populate(self, request, start, pages):
        c = self.c
        source = c.req_to_host_pool[request, start : start + len(pages)].to(torch.int64)
        stream = c.write_staging_stream
        stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(stream):
            for layer in range(c.mem_pool_device.layer_num):
                c.mem_pool_host.load_to_device_per_layer(
                    c.mem_pool_device, source, pages, layer, io_backend="kernel"
                )
            event = self.device_module.Event()
            event.record(stream)
            source.record_stream(stream)
            pages.record_stream(stream)
        return event

    def register_backed(self, request, imported_rows):
        if request not in self.residency.requests:
            self.residency.register(
                request,
                imported_rows,
                torch.empty(0, dtype=torch.int64, device=self.c.device),
                backed=True,
            )

    def admit_resident(self, req):
        """Keep imported full pages; back up only the mutable boundary eagerly."""
        c, rid = self.c, req.req_pool_idx
        if not self.config.resident_first:
            return False
        logical = c.mem_pool_device.translate_loc_from_full_to_compressed(
            c.req_to_token_pool.req_to_token[rid, : req.extend_range.end]
        )
        original = c.mem_pool_device.translate_loc_to_hisparse_device(logical).to(
            torch.int64
        )
        if not len(original):
            return False
        if not self.residency.reclaim(
            c.padded_buffer_size + self.config.admission_reserve_pages * c.page_size
        ):
            return False
        required = self.raw_allocator.alloc(c.padded_buffer_size)
        if required is None:
            return False
        complete = len(original) // c.page_size * c.page_size
        boundary = original[complete:]
        old_host_len = int(c.req_to_host_pool_allocated_len[rid])
        try:
            host = c.mem_pool_host.alloc_paged_token_slots(
                c.req_to_host_pool,
                c.req_to_host_pool_allocated_len,
                rid,
                0,
                len(original),
            )
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla

            # The newest row needs a canonical tail address even at an exact page boundary.
            transfer_cache_dsv4_mla(
                src_ptrs=c.mem_pool_device.data_ptrs,
                dst_ptrs=c.mem_pool_device.data_ptrs,
                src_indices=torch.cat((boundary, original[-1:])),
                dst_indices=torch.cat(
                    (
                        required[: len(boundary)],
                        required[c.device_buffer_size : c.device_buffer_size + 1],
                    )
                ),
            )
            self._backup(host[complete:], boundary)
            c.req_to_device_buffer[rid] = required
            c.req_device_buffer_size[rid] = len(required)
            c.req_device_buffer_token_locs[:, rid] = required.to(torch.int32)
            c.req_device_buffer_tokens[:, rid].fill_(-1)
            c.req_device_buffer_tokens[:, rid, : len(boundary)] = torch.arange(
                complete, len(original), dtype=torch.int32, device=c.device
            )
            c.req_device_buffer_tokens[:, rid, c.device_buffer_size] = len(original) - 1
        except BaseException:
            # The imported device mappings still own the original pages.
            self._retire(required)
            new_end = int(c.req_to_host_pool_allocated_len[rid])
            if new_end > old_host_len:
                new_pages = c.req_to_host_pool[rid, old_host_len:new_end]
                c.mem_pool_host.free(new_pages)
                new_pages.fill_(-1)
                c.req_to_host_pool_allocated_len[rid] = old_host_len
            c.req_to_device_buffer[rid].zero_()
            c.req_device_buffer_size[rid] = 0
            c.req_device_buffer_token_locs[:, rid].fill_(-1)
            c.req_device_buffer_tokens[:, rid].fill_(-1)
            raise

        c.mem_pool_device.full_to_hisparse_device_index_mapping[logical] = 0
        self._retire(boundary)
        self.residency.register(
            rid, len(original), original[:complete], backed=complete == 0
        )
        self.prepared_requests.add(rid)
        req.hisparse_staging = True
        event = self.device_module.Event()
        event.record()
        from sglang.srt.managers.hisparse_coordinator import HiSparseAct

        c.ack_staging_queue.append(HiSparseAct(event, event, req))
        return True

    def begin_forward(self):
        self.residency.refill()

    def abort_forward(self):
        self.device_module.current_stream().synchronize()
        if self.c.decode_producer_stream is not None:
            self.c.decode_producer_stream.synchronize()
        self.c.write_staging_stream.synchronize()
        self.pending.clear()

    def resolve(
        self,
        request_ids,
        lengths,
        selected,
        layer,
        *,
        output=None,
        num_real_requests=None,
    ):
        c, layout = self.c, self.layout
        slot = layout.layer_slots[layer]
        if any(layout.layer_slots[previous] == slot for previous in self.pending):
            raise RuntimeError(
                "Verification arena is still owned by an attention reader"
            )
        batch, width, k = selected.shape
        if (
            not 0 < width <= layout.max_verify_width
            or k != layout.top_k
            or batch > layout.num_requests
        ):
            raise ValueError("Verification demand exceeds the configured arena")
        # Flatten first so a narrower runtime window also gets a contiguous view.
        out = self.outputs[slot].view(-1)[: batch * width * k].view_as(selected)
        if output is not None and output.is_contiguous():
            out = output.view_as(selected)
        counts = self.io_counts[slot]
        if self.trace is not None:
            counts.zero_()
        args = dict(
            request_ids=request_ids,
            buffer_tokens=c.req_device_buffer_tokens[layer],
            buffer_locs=c.req_device_buffer_token_locs[layer],
            hot_ages=self.ages[layer],
            epochs=self.epochs[layer],
            device_buffer=c.mem_pool_device.kv_buffer[layer],
            staging_offset=c.mem_pool_device.dhisparse_staging_offsets[layer],
            staging_stride=layout.rows_per_request,
            hot_size=c.device_buffer_size,
            workspace=self.workspaces[slot],
            num_real_requests=(
                c.num_real_reqs if num_real_requests is None else num_real_requests
            ),
            counts=counts,
            item_bytes=c.item_size_bytes,
            packed=c.is_dsv4_hisparse,
        )
        host = c.mem_pool_host.kv_buffer[layer]
        with phase("swap_in_gather", layer=layer):
            kernels.resolve(
                selections=selected.to(torch.int32).contiguous(),
                seq_lens=lengths.to(torch.int32).contiguous(),
                optional_locs=self.optional_locs,
                host_locs=c.req_to_host_pool,
                host=host,
                output=out,
                **args,
            )
        with phase("swap_in_reread", layer=layer):
            if c.force_h2d_each_step:
                kernels.reread(
                    request_ids=request_ids,
                    host=host,
                    device_buffer=args["device_buffer"],
                    workspace=args["workspace"],
                    num_real_requests=args["num_real_requests"],
                    counts=counts,
                    staging_offset=args["staging_offset"],
                    staging_stride=layout.rows_per_request,
                    item_bytes=c.item_size_bytes,
                    packed=True,
                )
        self.pending[layer] = args
        if output is not None:
            if not output.is_contiguous():
                output.copy_(out.view_as(output))
            return output
        return out

    def finish_layer(self, layer):
        args = self.pending.pop(layer, None)
        if args is not None:
            with phase("swap_in_commit", layer=layer):
                kernels.promote(lru_slots=self.c.lru_slots[layer], **args)
            if self.trace is not None:
                self.trace.record_swap_in(
                    layer,
                    ("swap_in_gather", "swap_in_reread", "swap_in_commit"),
                    args["counts"],
                    self.c.force_h2d_each_step,
                    implementation="adaptive",
                )

    def release(self, request):
        self.residency.release(request)
        self.prepared_requests.discard(request)
        self.ages[:, request].zero_()
        self.epochs[:, request].zero_()

    def set_tail(self, requests, positions, locations):
        c = self.c
        c.req_device_buffer_tokens[:, requests, c.device_buffer_size] = positions.to(
            torch.int32
        )
        c.req_device_buffer_token_locs[:, requests, c.device_buffer_size] = (
            locations.to(torch.int32)
        )
