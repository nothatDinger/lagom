"""Shared verification materialization and dedicated full-step forecast slots."""
from __future__ import annotations

from typing import NamedTuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.srt.mem_cache.hisparse_staging import HiSparseStagingLayout
from sglang.srt.observability.lagom_perf import phase, prefetch_slot


class VerificationPlan(NamedTuple):
    table: torch.Tensor
    sources: torch.Tensor
    kinds: torch.Tensor
    demand: torch.Tensor
    tokens: torch.Tensor
    counts: torch.Tensor


@cache_once
def _module(hot_slots, width, top_k, forecast_capacity):
    args = make_cpp_args(hot_slots, width, top_k, forecast_capacity)
    return load_jit(
        "hisparse_staging", *args, cuda_files=["hisparse_staging.cuh"],
        cuda_wrappers=[
            ("plan_verify", f"staging::plan_verify<{args}>"),
            ("copy_verify_device", "staging::copy_verify<false>"),
            ("copy_verify_host", "staging::copy_verify<true>"),
            ("promote", f"staging::promote<{hot_slots}>"),
            ("prefetch", f"staging::prefetch<{hot_slots}, {max(forecast_capacity, 1)}>"),
        ],
    )


class HiSparseVerificationStaging:
    """One serial verification arena; N independent, event-protected forecasts.

    All allocations precede graph capture. Stable hot-cache locations never
    alias the arena or forecasts. Fixed per-layer descriptors prevent a later
    layer/batch from overwriting metadata still used by background transfers.
    """
    def __init__(self, *, layout: HiSparseStagingLayout, pool, hot_slots):
        self.layout, self.pool, self.hot_slots = layout, pool, hot_slots
        device = pool.device
        r, c = layout.request_slots, layout.verify_capacity
        h = 1 << (2 * c - 1).bit_length()

        def tensor(shape, dtype=torch.int32, fill=0):
            return torch.full(shape, fill, dtype=dtype, device=device)

        def payload(rows):
            return pool.create_buffer(num_pages=(rows + 63) // 64 + 1)

        self.arena = payload(r * c)
        self.plan = VerificationPlan(
            tensor((r, 2 * h)), tensor((r, c), torch.int64),
            tensor((r, c)), tensor((r, c)), tensor((r, c)), tensor((r,)),
        )
        self.req_ids = tensor((r,), torch.int64)
        self.history = tensor((r,))
        self.live = tensor((1,))
        self._active = None
        f = max(layout.forecast_capacity, 1)
        fh = 1 << (2 * f - 1).bit_length()
        self.forecasts = [payload(r * f) for _ in range(layout.slot_count)]
        self.forecast_indices = [tensor((r, 2 * fh), fill=-1) for _ in self.forecasts]
        self.forecast_sources = [tensor((r, f), torch.int64) for _ in self.forecasts]
        self.forecast_counts = [tensor((r,)) for _ in self.forecasts]
        self.forecast_tokens = [
            tensor((r, f), fill=-1)
            for _ in range(layout.sparse_layers if layout.prediction_steps else 0)
        ]
        self.forecast_reqs = [tensor((r,), torch.int64) for _ in self.forecast_tokens]
        self.forecast_live = [tensor((1,)) for _ in self.forecast_tokens]
        self._empty_index = tensor((r, 2), fill=-1)
        self.module = _module(hot_slots, layout.verify_width, layout.top_k, layout.forecast_capacity)

    def reset_request(self, request_slot):
        # Callers drain transfers before reset or host-page reclamation.
        for index, count in zip(self.forecast_indices, self.forecast_counts):
            index[request_slot].fill_(-1)
            count[request_slot].zero_()

    def resolve(
        self, *, layer_id, req_pool_indices, source_prefix_lens,
        compressed_seq_lens, top_k_result, window_locs, output_buffer,
        num_real_reqs, decode_width, host_locs, host_cache, device_cache,
        hot_tokens, hot_locs, state, prefetch_top_k=None,
    ):
        layout = self.layout
        batch = req_pool_indices.numel()
        shape = (batch, layout.verify_width, layout.top_k)
        if top_k_result.numel() != batch * layout.verify_capacity:
            raise ValueError("verification staging needs a fixed request-major window")
        self.req_ids[:batch].copy_(req_pool_indices)
        self.history[:batch].copy_(source_prefix_lens // 4)
        self.live.copy_(num_real_reqs)
        reqs, history = self.req_ids[:batch], self.history[:batch]
        lengths = compressed_seq_lens.reshape(batch, -1)
        if lengths.shape[1] == 1:
            lengths = lengths.expand(batch, layout.verify_width)
        lengths = lengths.to(torch.int32).contiguous()
        if lengths.shape != shape[:2]:
            raise ValueError("verification staging needs one causal length per query")
        raw = top_k_result.reshape(shape).to(torch.int32).contiguous()
        writer = window_locs.reshape(shape).to(torch.int32).contiguous()
        output_buffer.fill_(-1)
        slot = layout.slot_for_layer(layer_id) if layout.prediction_steps else None
        index = self.forecast_indices[slot] if slot is not None else self._empty_index
        forecast = self.forecasts[slot] if slot is not None else self.arena
        with phase("kv_plan", layer=layer_id):
            self.module.plan_verify(
                raw, lengths, writer, history, reqs, self.live,
                hot_tokens, hot_locs, host_locs,
                state.cache_index, state.cache_policy, state.scratch_state,
                index, *self.plan, output_buffer, decode_width,
            )
        with phase("kv_stage_copy", layer=layer_id):
            self.module.copy_verify_device(
                *self.plan, reqs, self.live, host_cache, device_cache, forecast, self.arena,
            )
        with phase("kv_refill", layer=layer_id):
            self.module.copy_verify_host(
                *self.plan, reqs, self.live, host_cache, device_cache, forecast, self.arena,
            )
        if layout.prediction_steps:
            if decode_width + layout.prediction_steps != layout.verify_width:
                raise ValueError("Lagom real/forecast windows differ from staging")
            future = raw[:, decode_width:].clone()
            future.masked_fill_((future < 0) | (future >= history[:, None, None]), -1)
            rank = layout.top_k if prefetch_top_k is None else prefetch_top_k
            future[:, :, rank:] = -1
            self.forecast_tokens[layer_id][:batch].copy_(
                future.reshape(batch, layout.forecast_capacity)
            )
            self.forecast_reqs[layer_id][:batch].copy_(reqs)
            self.forecast_live[layer_id].copy_(self.live)
        self._active = (
            layer_id, batch, host_locs, host_cache, device_cache,
            hot_tokens, hot_locs, state,
        )
        return output_buffer

    def attention_finished(self, layer_id):
        active = self._active
        if active is None or active[0] != layer_id:
            raise RuntimeError("verification arena released by the wrong attention layer")
        _, batch, host_locs, host_cache, device_cache, hot_tokens, hot_locs, state = active
        with phase("kv_promote", layer=layer_id):
            self.module.promote(
                *self.plan, self.req_ids[:batch], self.live, self.history[:batch],
                hot_tokens, hot_locs, state.cache_index, state.cache_policy,
                state.scratch_state, self.arena, device_cache,
            )
        self._active = None
        if not self.layout.prediction_steps:
            return batch, None
        slot = self.layout.slot_for_layer(layer_id)

        trace_slot = prefetch_slot(layer_id)

        def transfer():
            with phase("kv_prefetch", layer=layer_id, slot=trace_slot):
                self.module.prefetch(
                    self.forecast_tokens[layer_id][:batch],
                    self.forecast_reqs[layer_id][:batch], self.forecast_live[layer_id],
                    host_locs, self.forecast_indices[slot], self.forecast_sources[slot],
                    self.forecast_counts[slot], host_cache, self.forecasts[slot],
                    hot_tokens, state.cache_index, state.cache_policy, state.scratch_state,
                )

        return batch, transfer
