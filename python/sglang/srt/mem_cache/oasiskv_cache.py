"""Host-backed, head-wise OasisKV working sets.

This module has a CPU reference path and a CUDA path using the existing UVA
copy kernel. No full KV history is retained on the GPU. Native DSA may supply
logical candidates instead of using Quest summaries; ordinary MLA uses one
latent KV head, with its absorbed query and latent value dimensions.
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig


def quest_scores(query, key_min, key_max):
    """Return [KV heads, blocks] bounds; never average away opposing GQA heads."""
    heads, blocks, dim = key_min.shape
    if query.ndim != 2 or query.shape[1] != dim or query.shape[0] % heads:
        raise ValueError("Query heads/dimensions do not match the KV summaries")
    q = query.float().reshape(heads, -1, 1, dim)
    bounds = torch.maximum(q * key_min[:, None], q * key_max[:, None]).sum(-1)
    return bounds.amax(1)


@dataclass
class BlockPlan:
    block_ids: torch.Tensor
    last_selected: torch.Tensor
    sources: torch.Tensor
    destinations: torch.Tensor
    counts: torch.Tensor


def plan_delta(candidates, resident, last_selected, step, cap=None):
    """Retain hits and replace only capped LRU non-hits, entirely on device.

    Candidates are distinct logical block IDs, ranked in descending importance;
    -1 is padding. Plans have fixed width, with counts delimiting valid entries.
    """
    heads, slots = resident.shape
    if candidates.shape != resident.shape or last_selected.shape != resident.shape:
        raise ValueError("Candidates, resident IDs and ages must have equal shapes")
    matches = (candidates[:, :, None] == resident[:, None, :]) & (
        candidates[:, :, None] >= 0
    )
    candidate_hit = matches.any(-1)
    resident_hit = matches.any(1)
    missing = (candidates >= 0) & ~candidate_hit
    limit = slots if cap is None else min(slots, cap)
    counts = missing.sum(-1).clamp_max(limit).to(torch.int32)
    # Empty slots first, then least-recently-selected; hits cannot be evicted.
    priority = torch.where(resident < 0, -1, last_selected)
    priority = priority.masked_fill(resident_hit, torch.iinfo(torch.int64).max)
    targets = priority.argsort(dim=-1, stable=True)
    order = torch.arange(slots, device=resident.device).expand(heads, -1)
    compact_order = torch.where(missing, order, order + slots).argsort(
        dim=-1, stable=True
    )
    sources = candidates.gather(1, compact_order)
    admitted = order < counts[:, None]
    values = torch.where(admitted, sources, resident.gather(1, targets))
    updated = resident.scatter(1, targets, values)
    ages = torch.where(resident_hit, step, last_selected)
    ages = ages.scatter(
        1, targets, torch.where(admitted, step, ages.gather(1, targets))
    )
    return BlockPlan(updated, ages, sources, targets, counts)


class OasisKVPipeline:
    """Background selection/transfer; prediction and attention use the caller stream."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            self.select = torch.cuda.Stream(device=self.device)
            self.transfer = torch.cuda.Stream(device=self.device)

    def context(self, stage):
        return torch.cuda.stream(getattr(self, stage)) if self.cuda else nullcontext()


class OasisKVLayerCache:
    """One request, one layer. Host storage is [head, block, token, K+V]."""

    def __init__(self, config, heads, key_dim, value_dim, dtype, device, pipeline):
        self.config = config
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.dtype, self.device = dtype, torch.device(device)
        element_bytes = torch.empty((), dtype=dtype, device="cpu").element_size()
        # The UVA kernel copies aligned 128-bit records. Pad unusual head
        # dimensions too, rather than silently dropping a short byte tail.
        self.storage_dim = (
            ((key_dim + value_dim) * element_bytes + 15) // 16 * 16 // element_bytes
        )
        self.pipeline = pipeline
        self.length = 0
        self.step = 0
        self.capacity = 0
        self.host = None
        self.key_min = self.key_max = None
        self.selection_done = None
        self.pending = None
        self.backup_done = None
        self.refill_done = None
        self.initialized = False
        self.attention_blocks = None
        slots = config.history_blocks
        self.resident = torch.full((heads, slots), -1, dtype=torch.int64, device=device)
        self.ages = torch.full_like(self.resident, -1)
        self.data = torch.zeros(
            (heads, slots + 1, config.block_size, self.storage_dim),
            dtype=dtype,
            device=device,
        )
        self.real_heads = torch.tensor([heads], dtype=torch.int32, device=device)
        self.prefetch_submissions = 0

    def join(self, *, reclaim=False):
        # A forward may fail after selection but before transfer. Reclamation
        # and the next real-query refill must also join that metadata reader.
        for event in (
            self.selection_done,
            self.pending,
            self.backup_done,
            self.refill_done,
        ):
            if event is not None:
                if reclaim:
                    event.synchronize()
                else:
                    torch.cuda.current_stream(self.device).wait_event(event)
        if reclaim:
            self.selection_done = None
            self.pending = self.backup_done = None
            self.refill_done = None
        elif self.pipeline.cuda:
            self.resident.record_stream(torch.cuda.current_stream(self.device))
            self.ages.record_stream(torch.cuda.current_stream(self.device))

    def _grow(self, blocks):
        if blocks <= self.capacity:
            return
        # Host allocations may not be freed while UVA or D2H still accesses them.
        self.join(reclaim=True)
        capacity = max(blocks, max(4, self.capacity * 2))
        shape = (self.heads, capacity, self.config.block_size, self.storage_dim)
        host = torch.empty(
            shape, dtype=self.dtype, device="cpu", pin_memory=self.pipeline.cuda
        )
        lo = hi = None
        if not self.config.native_sparse:
            lo = torch.full(
                (self.heads, capacity, self.key_dim),
                float("inf"),
                dtype=torch.float32,
                device=self.device,
            )
            hi = torch.full_like(lo, -float("inf"))
        if self.capacity:
            host[:, : self.capacity].copy_(self.host)
            if lo is not None:
                lo[:, : self.capacity].copy_(self.key_min)
                hi[:, : self.capacity].copy_(self.key_max)
        self.host, self.key_min, self.key_max = host, lo, hi
        self.capacity = capacity

    def append(self, key, value, prefix):
        """Append committed tokens only. Draft KV never reaches this method."""
        if prefix != self.length:
            raise ValueError(f"OasisKV cache length {self.length} != prefix {prefix}")
        if key.shape[:2] != value.shape[:2] or key.shape[1:] != (
            self.heads,
            self.key_dim,
        ):
            raise ValueError("Invalid OasisKV append geometry")
        self.join()
        end = prefix + key.shape[0]
        block_size = self.config.block_size
        self._grow((end + block_size - 1) // block_size)
        packed = torch.cat((key, value), dim=-1).transpose(0, 1).contiguous()
        packed = F.pad(packed, (0, self.storage_dim - self.key_dim - self.value_dim))
        offset = 0
        while prefix + offset < end:
            pos = prefix + offset
            block, within = divmod(pos, block_size)
            count = min(block_size - within, end - pos)
            chunk = packed[:, offset : offset + count]
            self.host[:, block, within : within + count].copy_(chunk, non_blocking=True)
            if self.key_min is not None:
                k = chunk[..., : self.key_dim].float()
                self.key_min[:, block] = torch.minimum(
                    self.key_min[:, block], k.amin(1)
                )
                self.key_max[:, block] = torch.maximum(
                    self.key_max[:, block], k.amax(1)
                )
            # The last slot always contains the active block, outside historical LRU.
            if block == (end - 1) // block_size:
                if within == 0:
                    self.data[:, -1].zero_()
                self.data[:, -1, within : within + count].copy_(chunk)
            offset += count
        self.length = end
        if self.pipeline.cuda:
            self.backup_done = torch.cuda.Event()
            self.backup_done.record(torch.cuda.current_stream(self.device))

    def candidates(self, query, explicit=None, *, next_step=False):
        slots = self.config.history_blocks
        # The last block is already available in the dedicated active slot.
        complete = max(
            0, (self.length - (0 if next_step else 1)) // self.config.block_size
        )
        if explicit is not None:
            ids = explicit.reshape(self.heads, -1).to(torch.int64)
            ids = ids.masked_fill((ids < 0) | (ids >= complete), -1)
            # Native Indexer Top-K is distinct, and uses block_size=1.
            ids = ids[:, :slots]
            return F.pad(ids, (0, slots - ids.shape[1]), value=-1)
        if self.config.native_sparse:
            raise ValueError("Native DSA OasisKV requires logical Indexer Top-K")
        count = min(slots, complete)
        if not count:
            return torch.full_like(self.resident, -1)
        scores = quest_scores(
            query, self.key_min[:, :complete], self.key_max[:, :complete]
        )
        ids = scores.topk(count, dim=-1).indices
        return F.pad(ids, (0, slots - count), value=-1)

    def _copy_plan(self, plan):
        slots = self.config.history_blocks
        if self.pipeline.cuda:
            # This existing kernel reads pinned host pointers through CUDA UVA.
            # Treat one block-head K+V record as one opaque MLA copy item.
            from sglang.kernels.ops.kvcache.hisparse import copy_cache_planned_mla

            heads = torch.arange(self.heads, device=self.device)[:, None]
            src = (heads * self.capacity + plan.sources).to(torch.int64)
            dst = (heads * (slots + 1) + plan.destinations).to(torch.int32)
            copy_cache_planned_mla(
                miss_src=src,
                miss_dst=dst,
                miss_count=plan.counts,
                num_real_reqs=self.real_heads,
                host_cache=self.host,
                device_buffer=self.data,
                item_size_bytes=self.data[0, 0].numel() * self.data.element_size(),
                num_blocks=4,
            )
            for tensor in (src, dst, plan.counts):
                tensor.record_stream(torch.cuda.current_stream(self.device))
        else:
            for head, count in enumerate(plan.counts.tolist()):
                for i in range(count):
                    self.data[head, plan.destinations[head, i]].copy_(
                        self.host[head, plan.sources[head, i]]
                    )

    def seed(self, query, explicit=None):
        """Compatibility entry point for cold-start initialization."""
        if self.initialized:
            return
        self.prepare_attention(query, explicit)

    def prepare_attention(self, query, explicit=None):
        """Resolve every real-query miss on the foreground stream, without a cap.

        Draft predictions only warm the cache. They never decide which history
        the real attention uses. The same refill handles cold start and later
        prediction misses, using the UVA copy kernel for missing payloads only.
        """
        self.join()
        selected = self.candidates(query, explicit)
        plan = plan_delta(selected, self.resident, self.ages, self.step)
        try:
            self._copy_plan(plan)
            self.resident.copy_(plan.block_ids)
            self.ages.copy_(plan.last_selected)
        finally:
            # A forward can fail before background selection/transfer is
            # submitted. Reclamation must still wait for this UVA host reader.
            if self.pipeline.cuda:
                self.refill_done = torch.cuda.Event()
                self.refill_done.record(torch.cuda.current_stream(self.device))
        self.attention_blocks = selected
        self.initialized = True

    def predict_prefetch(self, draft_query, explicit=None):
        """Predict next-step candidates on the same stream as real attention."""
        return self.candidates(draft_query, explicit, next_step=True)

    def plan_prefetch(self, candidates):
        """Plan the next working set without publishing it to current attention.

        Call after prepare_attention(): the plan must observe the real query's
        refilled resident set. Updates allocate independent metadata tensors.
        """
        self.step += 1
        pipeline = self.pipeline
        if pipeline.cuda:
            # Observe prediction and real-query refill, but not attention.
            pipeline.select.wait_stream(torch.cuda.current_stream(self.device))
            candidates.record_stream(pipeline.select)
            self.resident.record_stream(pipeline.select)
            self.ages.record_stream(pipeline.select)
        with pipeline.context("select"):
            plan = plan_delta(
                candidates,
                self.resident,
                self.ages,
                self.step,
                self.config.max_new_blocks,
            )
            if pipeline.cuda:
                self.selection_done = torch.cuda.Event()
                self.selection_done.record(pipeline.select)
        return plan

    def submit_prefetch(self, plan):
        """After current attention's reads, submit the planned KV overwrites."""
        pipeline = self.pipeline
        if pipeline.cuda:
            pipeline.transfer.wait_event(self.selection_done)
            # Prediction and selection may finish before current attention.
            # Only the writer must wait until that attention has read data.
            pipeline.transfer.wait_stream(torch.cuda.current_stream(self.device))
            for tensor in (
                plan.sources,
                plan.destinations,
                plan.counts,
                plan.block_ids,
                plan.last_selected,
            ):
                tensor.record_stream(pipeline.transfer)
        with pipeline.context("transfer"):
            self._copy_plan(plan)
            self.resident.copy_(plan.block_ids)
            self.ages.copy_(plan.last_selected)
            if pipeline.cuda:
                self.pending = torch.cuda.Event()
                self.pending.record(pipeline.transfer)
        # Keep current-round logical IDs unchanged until attention is enqueued.
        # The next append/attention joins pending before using these new IDs.
        self.prefetch_submissions += 1

    def prefetch(self, draft_query, explicit=None):
        """Convenience path for callers whose current attention is already done."""
        candidates = self.predict_prefetch(draft_query, explicit)
        self.submit_prefetch(self.plan_prefetch(candidates))

    def attend(self, query, scale, *, draft_key=None, draft_value=None, logit_cap=0):
        """Both queries see the same resident history; only draft sees draft KV."""
        count, q_heads, _ = query.shape
        group = q_heads // self.heads
        active = (self.length - 1) // self.config.block_size
        block_ids = torch.cat(
            (self.resident, self.resident.new_full((self.heads, 1), active)), 1
        )
        positions = block_ids[..., None] * self.config.block_size + torch.arange(
            self.config.block_size, device=self.device
        )
        valid = (block_ids[..., None] >= 0) & (positions < self.length)
        if self.attention_blocks is not None:
            # Unselected prefetched entries may remain in otherwise unused
            # slots. They must not expand a padded/small real Top-K selection.
            selected = (
                self.resident[:, :, None] == self.attention_blocks[:, None, :]
            ).any(-1)
            valid[:, :-1] &= selected[..., None]
        packed = self.data.flatten(1, 2)
        key, value = (
            packed[..., : self.key_dim],
            packed[..., self.key_dim : self.key_dim + self.value_dim],
        )
        mask = valid.flatten(1).unsqueeze(1).expand(-1, count, -1)
        if draft_key is not None:
            key = torch.cat((key, draft_key[:, None]), 1)
            value = torch.cat((value, draft_value[:, None]), 1)
            draft_mask = torch.zeros(
                (self.heads, count, 1), dtype=torch.bool, device=self.device
            )
            draft_mask[:, -1] = True
            mask = torch.cat((mask, draft_mask), -1)
        q = query.transpose(0, 1).reshape(self.heads, group, count, self.key_dim)
        key = key[:, None].expand(-1, group, -1, -1)
        value = value[:, None].expand(-1, group, -1, -1)
        if logit_cap:
            logits = torch.matmul(q.float(), key.float().transpose(-1, -2)) * scale
            logits = logit_cap * torch.tanh(logits / logit_cap)
            weights = logits.masked_fill(~mask[:, None], -float("inf")).softmax(-1)
            out = torch.matmul(weights, value.float()).to(query.dtype)
        else:
            out = F.scaled_dot_product_attention(
                q, key, value, attn_mask=mask[:, None], scale=scale
            )
        return out.reshape(q_heads, count, self.value_dim).transpose(0, 1).contiguous()

    def dense_prefill(self, query, prefix, scale, *, indices=None, logit_cap=0):
        """Chunked causal prefill; full KV is transient on GPU for this layer only."""
        if self.backup_done is not None:
            self.backup_done.synchronize()
        n = self.length
        packed = self.host.flatten(1, 2)[:, :n].to(self.device, non_blocking=True)
        key, value = (
            packed[..., : self.key_dim],
            packed[..., self.key_dim : self.key_dim + self.value_dim],
        )
        group = query.shape[1] // self.heads
        key = key[:, None].expand(-1, group, -1, -1)
        value = value[:, None].expand(-1, group, -1, -1)
        results = []
        for start in range(0, query.shape[0], self.config.prefill_chunk_size):
            chunk = query[start : start + self.config.prefill_chunk_size]
            q = chunk.transpose(0, 1).reshape(self.heads, group, -1, self.key_dim)
            mask = torch.arange(n, device=self.device)[None] <= (
                prefix
                + start
                + torch.arange(chunk.shape[0], device=self.device)[:, None]
            )
            if indices is not None:
                selected = indices[start : start + chunk.shape[0]]
                # Padding at -1 must not overwrite a genuine selection of 0.
                valid = (selected >= 0) & (selected < n)
                selected_mask = torch.zeros_like(mask, dtype=torch.int32)
                selected_mask.scatter_add_(
                    1, selected.clamp(0, n - 1).long(), valid.int()
                )
                mask &= selected_mask > 0
            if logit_cap:
                logits = torch.matmul(q.float(), key.float().transpose(-1, -2)) * scale
                logits = logit_cap * torch.tanh(logits / logit_cap)
                weights = logits.masked_fill(~mask, -float("inf")).softmax(-1)
                out = torch.matmul(weights, value.float()).to(query.dtype)
            else:
                out = F.scaled_dot_product_attention(
                    q, key, value, attn_mask=mask, scale=scale
                )
            results.append(
                out.reshape(query.shape[1], chunk.shape[0], self.value_dim).transpose(
                    0, 1
                )
            )
        return torch.cat(results, 0)


class OasisKVCache:
    """Request generation ownership shared by all target layers and backends."""

    def __init__(self, config: OasisKVConfig, device):
        self.config = config
        self.pipeline = OasisKVPipeline(device)
        self.layers = {}

    def get(self, slot, generation, layer_id, key, value):
        identity = (slot, generation, layer_id)
        if identity not in self.layers:
            for old in list(self.layers):
                if old[0] == slot and old[1] != generation:
                    self.layers.pop(old).join(reclaim=True)
            self.layers[identity] = OasisKVLayerCache(
                self.config,
                key.shape[1],
                key.shape[2],
                value.shape[2],
                key.dtype,
                key.device,
                self.pipeline,
            )
        return self.layers[identity]

    def release(self, slot):
        for identity in list(self.layers):
            if identity[0] == slot:
                self.layers.pop(identity).join(reclaim=True)

    def clear(self):
        for layer in self.layers.values():
            layer.join(reclaim=True)
        self.layers.clear()
