"""Adaptive DHiSparse serving state, shared arenas and residency lifecycle."""

from __future__ import annotations

from functools import wraps

import torch

from sglang.kernels.ops.kvcache import dhisparse as kernels
from sglang.srt.mem_cache.dhisparse import RequiredPageAllocator, ResidencyManager
from sglang.srt.utils import get_device_module


def finish_dhisparse_attention(function):
    """Keep promotion inside graph capture and after the attention launch."""

    @wraps(function)
    def wrapped(self, q, k, v, layer, forward_batch, *args, **kwargs):
        result = function(self, q, k, v, layer, forward_batch, *args, **kwargs)
        coordinator = getattr(self, "hisparse_coordinator", None)
        adaptive = getattr(coordinator, "adaptive", None)
        if adaptive is not None:
            if coordinator.is_dsv4_hisparse:
                ratio, source, _ = self.token_to_kv_pool.layer_mapping[layer.layer_id]
                if ratio == 4:
                    adaptive.finish_layer(source)
            else:
                adaptive.finish_layer(
                    layer.layer_id - coordinator.mem_pool_device.start_layer
                )
        return result

    return wrapped


def rollback_dhisparse_on_error(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except BaseException:
            c = getattr(self.model_runner, "hisparse_coordinator", None)
            if getattr(c, "adaptive", None) is not None:
                c.adaptive.abort_forward()
                c.adaptive.rollback_verify()
                if c._active_dspark_window is not None:
                    c.rollback_dspark_verify_window(c._active_dspark_window)
            raise

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
        self.ages = torch.zeros((layers, n, hot), dtype=torch.int64, device=c.device)
        self.epochs = torch.zeros((layers, n), dtype=torch.int64, device=c.device)
        occurrences = self.layout.max_verify_width * self.layout.top_k
        self.workspaces = [
            kernels.allocate_workspace(n, occurrences, hot, c.device)
            for _ in range(self.layout.arenas)
        ]
        self.outputs = [
            torch.full((n, occurrences), -1, dtype=torch.int32, device=c.device)
            for _ in range(self.layout.arenas)
        ]
        self.pending = {}
        self.prefetched = {}
        self.prepared_requests = set()
        self.active_verify = None
        self.prefetch_stream = self.device_module.Stream()
        self.events = [self.device_module.Event() for _ in range(layers)]
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
        self.prefetch_stream.synchronize()
        if self.c.decode_producer_stream is not None:
            self.c.decode_producer_stream.synchronize()
        self.device_module.current_stream().synchronize()
        if len(pages):
            self.raw_allocator.free(pages)

    def _backup(self, host, device):
        c = self.c
        if not len(host):
            return
        if c.is_dsv4_hisparse:
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla

            transfer_cache_dsv4_mla(
                src_ptrs=c.mem_pool_device.data_ptrs,
                dst_ptrs=c.mem_pool_host.data_ptrs,
                src_indices=device.to(torch.int64),
                dst_indices=host.to(torch.int64),
            )
        else:
            c.mem_pool_host.backup_from_device_all_layer(
                c.mem_pool_device, host, device, io_backend="kernel"
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
        if not self.residency.reclaim(c.padded_buffer_size):
            return False
        required = self.raw_allocator.alloc(c.padded_buffer_size)
        if required is None:
            return False
        try:
            host = c.mem_pool_host.alloc_paged_token_slots(
                c.req_to_host_pool,
                c.req_to_host_pool_allocated_len,
                rid,
                0,
                len(original),
            )
        except BaseException:
            self.raw_allocator.free(required)
            raise
        complete = len(original) // c.page_size * c.page_size
        boundary = original[complete:]
        c.req_to_device_buffer[rid] = required
        c.req_device_buffer_size[rid] = len(required)
        c.req_device_buffer_token_locs[:, rid] = required.to(torch.int32)
        c.req_device_buffer_tokens[:, rid].fill_(-1)
        # Copy boundary rows to stable hot slots; the newest row also has the
        # canonical tail address expected by the decode writer.
        sources = torch.cat((boundary, original[-1:]))
        destinations = torch.cat(
            (
                required[: len(boundary)],
                required[c.device_buffer_size : c.device_buffer_size + 1],
            )
        )
        if c.is_dsv4_hisparse:
            from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla

            transfer_cache_dsv4_mla(
                src_ptrs=c.mem_pool_device.data_ptrs,
                dst_ptrs=c.mem_pool_device.data_ptrs,
                src_indices=sources,
                dst_indices=destinations,
            )
        else:
            c.mem_pool_device.transfer_values_on_device(destinations, sources)
        c.req_device_buffer_tokens[:, rid, : len(boundary)] = torch.arange(
            complete, len(original), dtype=torch.int32, device=c.device
        )
        c.req_device_buffer_tokens[:, rid, c.device_buffer_size] = len(original) - 1
        self._backup(host[complete:], boundary)
        c.mem_pool_device.full_to_hisparse_device_index_mapping[logical] = 0
        if len(boundary):
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
        self.prefetch_stream.synchronize()
        self.pending.clear()
        self.prefetched.clear()

    def resolve(self, request_ids, lengths, selected, layer, *, output=None):
        c, layout = self.c, self.layout
        if layer in self.prefetched:
            self.events[layer].wait(self.device_module.current_stream())
            result = self.prefetched.pop(layer)
            if output is not None:
                output.copy_(result.view_as(output))
                return output
            return result
        slot = layout.layer_slots[layer]
        if layer in self.pending:
            self.finish_layer(layer)
        for previous in tuple(self.pending):
            if layout.layer_slots[previous] == slot:
                self.finish_layer(previous)
        batch, width, k = selected.shape
        out = self.outputs[slot][:batch, : width * k].view(batch, width, k)
        # Width may be smaller than maximum; use contiguous capture-owned views.
        if not out.is_contiguous():
            out = torch.empty_like(selected, dtype=torch.int32)
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
            num_real_requests=c.num_real_reqs,
            item_bytes=c.item_size_bytes,
            packed=c.is_dsv4_hisparse,
        )
        kernels.resolve(
            selections=selected.to(torch.int32).contiguous(),
            seq_lens=lengths.contiguous(),
            optional_locs=self.optional_locs,
            host_locs=c.req_to_host_pool,
            host=c.mem_pool_host.kv_buffer[layer],
            output=out,
            **args,
        )
        self.pending[layer] = args
        group = c._prefetch_groups.get(layer, [])
        if group:
            self.prefetch_stream.wait_stream(self.device_module.current_stream())
            with self.device_module.stream(self.prefetch_stream):
                for follower in group:
                    self.prefetched[follower] = self.resolve(
                        request_ids, lengths, selected, follower
                    )
                    self.events[follower].record(self.prefetch_stream)
        if output is not None:
            output.copy_(out.view_as(output))
            return output
        return out

    def finish_layer(self, layer):
        args = self.pending.pop(layer, None)
        if args is not None:
            kernels.promote(**args)

    def release(self, request):
        self.residency.release(request)
        self.prepared_requests.discard(request)
        self.ages[:, request].zero_()
        self.epochs[:, request].zero_()

    def set_tail(self, requests, positions, locations, *, offsets=None):
        c = self.c
        if offsets is None:
            offsets = torch.full_like(positions, c.device_buffer_size)
        c.req_device_buffer_tokens[:, requests, offsets] = positions.to(torch.int32)
        c.req_device_buffer_token_locs[:, requests, offsets] = locations.to(torch.int32)

    def prepare_verify(self, request_ids, prefix_lens, logical_locs):
        """Transactional uncompressed target writes for a linear draft block."""
        c = self.c
        if c.is_dsv4_hisparse:
            raise ValueError("Compressed target uses the C4 transaction")
        if self.active_verify is not None:
            raise RuntimeError("Unfinished DHiSparse target verification")
        batch, width = logical_locs.shape
        if width + 1 > c.padded_buffer_size - c.device_buffer_size:
            raise ValueError("Verify block exceeds reserved tail")
        c._eager_backup_previous_token(
            prefix_lens + 1,
            request_ids,
            prefix_lens.to("cpu") + 1,
            request_ids.to("cpu"),
        )
        c.wait_for_pending_backup()
        mapping = c.mem_pool_device.full_to_hisparse_device_index_mapping
        previous = mapping[logical_locs].clone()
        offsets = torch.arange(width, device=c.device)
        tail_slots = c.device_buffer_size + 1 + offsets
        physical = c.req_to_device_buffer[request_ids[:, None], tail_slots[None, :]]
        positions = prefix_lens[:, None] + offsets[None, :]
        mapping[logical_locs] = physical
        c.req_device_buffer_tokens[:, request_ids[:, None], tail_slots[None, :]] = (
            positions.to(torch.int32)
        )
        c.req_device_buffer_token_locs[:, request_ids[:, None], tail_slots[None, :]] = (
            physical.to(torch.int32)
        )
        self.active_verify = (
            request_ids,
            prefix_lens,
            logical_locs,
            previous,
            physical,
            tail_slots,
        )

    def rollback_verify(self):
        if self.active_verify is None:
            return
        request_ids, _, logical, previous, _, tail_slots = self.active_verify
        c = self.c
        c.mem_pool_device.full_to_hisparse_device_index_mapping[logical] = previous
        c.req_device_buffer_tokens[:, request_ids[:, None], tail_slots[None, :]] = -1
        self.active_verify = None

    def commit_verify(self, commit_lens):
        if self.active_verify is None:
            return
        c = self.c
        request_ids, prefixes, logical, previous, physical, _ = self.active_verify
        commits = commit_lens.to("cpu").tolist()
        reqs, starts = request_ids.to("cpu").tolist(), prefixes.to("cpu").tolist()
        try:
            for row, (rid, start, count) in enumerate(zip(reqs, starts, commits)):
                if not 0 <= count <= physical.shape[1]:
                    raise ValueError("Accepted prefix exceeds the verified block")
                if not count:
                    continue
                host = c.mem_pool_host.alloc_paged_token_slots(
                    c.req_to_host_pool,
                    c.req_to_host_pool_allocated_len,
                    rid,
                    start,
                    count,
                )
                self._backup(host, physical[row, :count])
                # Copy the final committed row back before scratch reuse.
                destination = c.req_to_device_buffer[
                    rid, c.device_buffer_size : c.device_buffer_size + 1
                ]
                c.mem_pool_device.transfer_values_on_device(
                    destination, physical[row, count - 1 : count]
                )
                c.req_device_buffer_tokens[:, rid, c.device_buffer_size] = (
                    start + count - 1
                )
                c.req_device_buffer_token_locs[:, rid, c.device_buffer_size] = (
                    destination[0].to(torch.int32)
                )
            c._backup_done_event.record(self.device_module.current_stream())
            c._has_pending_backup = True
        finally:
            self.rollback_verify()
