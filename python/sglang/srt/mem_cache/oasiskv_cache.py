"""Host-backed, head-wise OasisKV working sets.

Dense and sparse requests use a common physical page pool. Full history stays
on the GPU until the sparse threshold; longer requests use pinned host backing.
This module also retains a CPU reference path. Native DSA may supply
logical candidates instead of using Quest summaries; ordinary MLA uses one
latent KV head, with its absorbed query and latent value dimensions.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from weakref import WeakSet

import torch
import torch.nn.functional as F
from sglang.srt.mem_cache.oasiskv_config import OasisKVConfig
from sglang.srt.mem_cache.oasiskv_pages import OasisKVPagePool, OasisKVPageTable
from sglang.srt.mem_cache.oasiskv_prefill import prefill_attention


def quest_scores(query, key_min, key_max):
    """Return [KV heads, blocks] bounds; never average away opposing GQA heads."""
    heads, _blocks, dim = key_min.shape
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

    def __init__(
        self, config, heads, key_dim, value_dim, dtype, device, pipeline, page_pool=None
    ):
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
        self.read_done = None
        self.initialized = False
        self.attention_blocks = None
        self._prefetch_base = None
        slots = config.history_blocks
        self.resident = torch.full((heads, slots), -1, dtype=torch.int64, device=device)
        self.ages = torch.full_like(self.resident, -1)
        self.page_table = OasisKVPageTable(page_pool, slots) if page_pool else None
        self.sparse = page_pool is None or config.native_sparse
        self._data = (
            None
            if page_pool
            else torch.zeros(
                (heads, slots + 1, config.block_size, self.storage_dim),
                dtype=dtype,
                device=device,
            )
        )
        self.real_heads = torch.tensor([heads], dtype=torch.int32, device=device)
        self.prefetch_submissions = 0
        self._graph_bindings = WeakSet()

    def _invalidate_graph_bindings(self, *, release=False):
        for binding in tuple(self._graph_bindings):
            binding.invalidate(release=release)

    @property
    def data(self):
        # Tensor reference/legacy attention only. CUDA kernels dereference
        # the stable page table and never materialize a contiguous working set.
        return self.page_table.materialize() if self.page_table else self._data

    def block(self, slot):
        return self.page_table.block(slot) if self.page_table else self._data[:, slot]

    def graph_row(self, prefix):
        return [
            0 if self.host is None else self.host.data_ptr(),
            0 if self.page_table else self._data.data_ptr(),
            0 if self.key_min is None else self.key_min.data_ptr(),
            0 if self.key_max is None else self.key_max.data_ptr(),
            self.resident.data_ptr(),
            self.ages.data_ptr(),
            self.capacity,
            prefix,
            self.step,
            self.page_table.pointers.data_ptr() if self.page_table else 0,
            int(not self.sparse),
        ]

    def prepare_append(self, end):
        self._invalidate_graph_bindings()
        was_dense = not self.sparse
        transition = was_dense and end > self.config.token_budget
        if transition:
            self.join()
            self.sparse = True
        self._grow((end + self.config.block_size - 1) // self.config.block_size)
        if self.page_table is None:
            return
        if transition and self.host is None:
            self.host = torch.empty(
                (self.heads, self.capacity, self.config.block_size, self.storage_dim),
                dtype=self.dtype,
                pin_memory=self.pipeline.cuda,
            )
        old_active = self.page_table.active
        if transition and self.length:
            for block in range(
                (self.length + self.config.block_size - 1) // self.config.block_size
            ):
                slot = self.config.history_blocks if block == old_active else block
                self.host[:, block].copy_(
                    self.block(slot), non_blocking=self.pipeline.cuda
                )
        active = max(0, (end - 1) // self.config.block_size)
        if transition or active != old_active:
            if self.page_table.needs_update(end, sparse=self.sparse):
                self.join()
            self.page_table.ensure(end, sparse=self.sparse)
            if was_dense:
                complete = min(
                    active if not transition else self.length // self.config.block_size,
                    self.config.history_blocks,
                )
                ids = torch.arange(self.config.history_blocks, device=self.device)
                self.resident.copy_(
                    torch.where(ids < complete, ids, -1).expand(self.heads, -1)
                )
        if transition and self.pipeline.cuda:
            self.backup_done = torch.cuda.Event()
            self.backup_done.record()

    def release(self):
        self.join(reclaim=True)
        self._invalidate_graph_bindings(release=True)
        if self.page_table:
            self.page_table.release()

    def join(self, *, reclaim=False):
        # A forward may fail after selection but before transfer. Reclamation
        # and the next real-query refill must also join that metadata reader.
        for event in (
            self.selection_done,
            self.pending,
            self.backup_done,
            self.refill_done,
            self.read_done,
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
            self.read_done = None
        elif self.pipeline.cuda:
            self.resident.record_stream(torch.cuda.current_stream(self.device))
            self.ages.record_stream(torch.cuda.current_stream(self.device))

    def _grow(self, blocks):
        if blocks <= self.capacity:
            return
        self._invalidate_graph_bindings()
        # Host allocations may not be freed while UVA or D2H still accesses them.
        self.join(reclaim=True)
        capacity = max(blocks, max(4, self.capacity * 2))
        shape = (self.heads, capacity, self.config.block_size, self.storage_dim)
        host = (
            None
            if not self.sparse
            else torch.empty(
                shape, dtype=self.dtype, device="cpu", pin_memory=self.pipeline.cuda
            )
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
            if self.host is not None:
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
        self._prefetch_base = None
        self.prepare_append(end)
        packed = torch.cat((key, value), dim=-1).transpose(0, 1).contiguous()
        packed = F.pad(packed, (0, self.storage_dim - self.key_dim - self.value_dim))
        if self.pipeline.cuda and key.shape[0]:
            self._append_batched(packed, prefix)
        else:
            self._append_reference(packed, prefix)
        self.length = end
        if self.pipeline.cuda:
            self.backup_done = torch.cuda.Event()
            self.backup_done.record(torch.cuda.current_stream(self.device))

    def _append_batched(self, packed, prefix):
        """CUDA chunk write: one summary/page kernel and one DMA per KV head."""
        import triton
        import triton.language as tl
        from sglang.srt.layers.attention.triton_ops import oasiskv as kernels

        count = packed.shape[1]
        end = prefix + count
        block = self.config.block_size
        tile = min(triton.next_power_of_2(self.storage_dim), 128)
        blocks = (
            1
            if self.config.native_sparse and self.sparse
            else triton.cdiv(end, block) - prefix // block
        )
        kernels.append_chunk[(self.heads, blocks, triton.cdiv(self.storage_dim, tile))](
            packed,
            self.key_min if self.key_min is not None else packed,
            self.key_max if self.key_max is not None else packed,
            self.page_table.pointers if self.page_table else packed,
            self._data if self._data is not None else packed,
            prefix,
            count,
            self.capacity,
            KD=self.key_dim,
            STORAGE=self.storage_dim,
            SLOTS=self.config.history_blocks,
            BLOCK=block,
            POOLED=self.page_table is not None,
            DENSE=not self.sparse,
            NATIVE=self.config.native_sparse,
            DTYPE={
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }[self.dtype],
            ROWS=triton.next_power_of_2(block),
            TILE=tile,
            num_warps=4,
        )
        if self.host is not None:
            # Each view is contiguous. A single strided [heads,tokens,storage]
            # D2H copy can require temporary packing and CPU-side scatter.
            host = self.host.view(self.heads, self.capacity * block, self.storage_dim)
            for head in range(self.heads):
                host[head, prefix:end].copy_(packed[head], non_blocking=True)

    def _append_reference(self, packed, prefix):
        """CPU reference and GPU A/B benchmark of the former per-block path."""
        end = prefix + packed.shape[1]
        block_size = self.config.block_size
        offset = 0
        while prefix + offset < end:
            pos = prefix + offset
            block, within = divmod(pos, block_size)
            count = min(block_size - within, end - pos)
            chunk = packed[:, offset : offset + count]
            if self.host is not None:
                self.host[:, block, within : within + count].copy_(
                    chunk, non_blocking=True
                )
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
                    self.block(self.config.history_blocks).zero_()
                self.block(self.config.history_blocks)[
                    :, within : within + count
                ].copy_(chunk)
            elif not self.sparse:
                self.block(block)[:, within : within + count].copy_(chunk)
            offset += count

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
        if self.page_table and self.pipeline.cuda:
            import triton
            import triton.language as tl
            from sglang.srt.layers.attention.triton_ops import oasiskv as kernels

            table = torch.tensor(
                [self.graph_row(self.length - 1)], dtype=torch.int64, device=self.device
            )
            kernels.copy_misses[(self.heads, 4)](
                table,
                plan.sources,
                plan.destinations,
                plan.counts,
                GROUPS=self.heads,
                HEADS=self.heads,
                SLOTS=slots,
                BLOCK=self.config.block_size,
                STORAGE=self.storage_dim,
                DTYPE={
                    torch.float16: tl.float16,
                    torch.bfloat16: tl.bfloat16,
                    torch.float32: tl.float32,
                }[self.dtype],
                TILE=min(
                    triton.next_power_of_2(self.config.block_size * self.storage_dim),
                    1024,
                ),
                WORKERS=4,
                num_warps=4,
            )
            return
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
                    self.block(int(plan.destinations[head, i]))[head].copy_(
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
        self._invalidate_graph_bindings()
        self.join()
        self._prefetch_base = None
        if not self.sparse:
            self.attention_blocks = self.resident.clone()
            self.initialized = True
            return
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
        # Keep the completed real-query working set as this step's prefetch
        # base. In particular, freshly refilled blocks are hits, not new I/O.
        self._prefetch_base = (plan.block_ids, plan.last_selected)
        self.initialized = True

    def predict_prefetch(self, draft_query, explicit=None):
        """Predict next-step candidates on the same stream as real attention."""
        if not self.sparse:
            return torch.full_like(self.resident, -1)
        return self.candidates(draft_query, explicit, next_step=True)

    def plan_prefetch(self, candidates):
        """Plan the next working set without publishing it to current attention.

        Call after prepare_attention(): the plan must observe the real query's
        refilled resident set. Updates allocate independent metadata tensors.
        """
        self._invalidate_graph_bindings()
        self.step += 1
        pipeline = self.pipeline
        resident, ages = self._prefetch_base or (self.resident, self.ages)
        self._prefetch_base = None
        if pipeline.cuda:
            # Observe prediction and real-query refill, but not attention.
            pipeline.select.wait_stream(torch.cuda.current_stream(self.device))
            candidates.record_stream(pipeline.select)
            resident.record_stream(pipeline.select)
            ages.record_stream(pipeline.select)
        with pipeline.context("select"):
            plan = plan_delta(
                candidates,
                resident,
                ages,
                self.step,
                self.config.max_new_blocks,
            )
            if pipeline.cuda:
                self.selection_done = torch.cuda.Event()
                self.selection_done.record(pipeline.select)
        return plan

    def submit_prefetch(self, plan):
        """After current attention's reads, submit the planned KV overwrites."""
        self._invalidate_graph_bindings()
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
        if self.page_table and self.pipeline.cuda:
            return self._finish_attention(
                self._attend_paged(query, scale, draft_key, draft_value, logit_cap)
            )
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
        return self._finish_attention(
            out.reshape(q_heads, count, self.value_dim).transpose(0, 1).contiguous()
        )

    def _finish_attention(self, output):
        # Page reclamation must cover readers too, including dense prefill and
        # unpaired eager decode where no subsequent prefetch records a fence.
        if self.device.type == "cuda":
            self.read_done = torch.cuda.Event()
            self.read_done.record(torch.cuda.current_stream(self.device))
        return output

    def _attend_paged(self, query, scale, draft_key, draft_value, logit_cap):
        """Eager decode uses the same paged, paired kernel as full graph decode."""
        import triton
        from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
        from sglang.srt.mem_cache.oasiskv_attention import (
            GroupedAttentionWorkspace,
            use_grouped_attention,
        )

        width, qheads, _ = query.shape
        if width not in (1, 2) or (width == 2) != (draft_key is not None):
            raise ValueError("Paged decode expects one real query or a real/draft pair")
        slots, block = self.config.history_blocks, self.config.block_size
        table = torch.tensor(
            [self.graph_row(self.length - 1)], dtype=torch.int64, device=self.device
        )
        selected = self.resident >= 0
        if self.attention_blocks is not None:
            selected &= (
                self.resident[:, :, None] == self.attention_blocks[:, None, :]
            ).any(-1)
        key = torch.empty(
            (width, self.heads, self.key_dim), dtype=self.dtype, device=self.device
        )
        value = torch.empty(
            (width, self.heads, self.value_dim), dtype=self.dtype, device=self.device
        )
        if width == 2:
            key[1].copy_(draft_key)
            value[1].copy_(draft_value)
        if query.dtype == self.dtype and use_grouped_attention(
            self.config, self.dtype, qheads // self.heads, self.key_dim, self.value_dim
        ):
            workspace = GroupedAttentionWorkspace(
                1,
                self.heads,
                qheads // self.heads,
                self.key_dim,
                self.value_dim,
                self.config,
                self.device,
            )
            workspace.prepare(
                table, self.resident, selected, self.storage_dim, self.dtype
            )
            output = query.new_empty((width, qheads, self.value_dim))
            workspace.forward(
                query.contiguous(),
                key,
                value,
                output,
                batch=1,
                width=width,
                storage=self.storage_dim,
                scale=scale,
                cap=logit_cap,
            )
            return output
        tokens = (slots + 1) * block + 1
        splits = min(8, triton.cdiv(tokens, 256))
        partial = torch.empty(
            (width, qheads, splits, self.value_dim + 2),
            dtype=torch.float32,
            device=self.device,
        )
        output = query.new_empty((width, qheads, self.value_dim))
        kernels.attention_partial[(1, qheads, splits)](
            table,
            query.contiguous(),
            key,
            value,
            selected,
            partial,
            HEADS=self.heads,
            GROUP=qheads // self.heads,
            KD=self.key_dim,
            VD=self.value_dim,
            STORAGE=self.storage_dim,
            SLOTS=slots,
            BLOCK=block,
            WIDTH=width,
            SCALE=scale,
            CAP=logit_cap,
            SPLITS=splits,
            PER_SPLIT=triton.cdiv(tokens, splits * 32) * 32,
            DK=triton.next_power_of_2(self.key_dim),
            DV=triton.next_power_of_2(self.value_dim),
            TILE=32,
        )
        kernels.attention_reduce[(width, qheads)](
            partial,
            output,
            QHEADS=qheads,
            VD=self.value_dim,
            SPLITS=splits,
            NS=triton.next_power_of_2(splits),
            DV=triton.next_power_of_2(self.value_dim),
        )
        return output

    def _prefill_prefix(self, prefix):
        """Read only old history; never inspect pinned payloads on the CPU."""
        if self.host is None:
            active = (prefix - 1) // self.config.block_size
            return torch.cat(
                [
                    self.block(i if i < active else self.config.history_blocks)
                    for i in range(active + 1)
                ],
                1,
            )[:, :prefix]
        packed = torch.empty(
            (self.heads, prefix, self.storage_dim),
            dtype=self.dtype,
            device=self.device,
        )
        # Each source is contiguous pinned memory. Copying a strided multi-head
        # CPU view can first pack it on the CPU, which would require a host-side
        # wait for D2H. These DMA reads instead follow join()'s stream fences.
        try:
            for head in range(self.heads):
                packed[head].copy_(
                    self.host[head].flatten(0, 1)[:prefix], non_blocking=True
                )
        finally:
            # Reclamation must also cover an exception in the subsequent
            # attention call, before append() records its normal completion.
            if self.pipeline.cuda:
                self.read_done = torch.cuda.Event()
                self.read_done.record(torch.cuda.current_stream(self.device))
        return packed

    def prefill(self, query, key, value, prefix, scale, *, indices=None, logit_cap=0):
        """Attend to current GPU K/V before backing them up for sparse decode."""
        if prefix != self.length or query.shape[0] != key.shape[0]:
            raise ValueError(
                "OasisKV prefill prefix/chunk does not match cache history"
            )
        self.join()
        # A dense prefix is already on the GPU. Read it before prepare_append
        # changes its active-page mapping or transitions it to host backing.
        history = self._prefill_prefix(prefix) if prefix and self.host is None else None
        # Allocate/grow before attention. Otherwise _grow's reclamation fence
        # could wait for the attention we just launched on every new request.
        self.prepare_append(prefix + key.shape[0])
        if prefix and history is None:
            history = self._prefill_prefix(prefix)
        if history is None:
            full_key, full_value = key, value
        else:
            full_key = torch.cat((history[..., : self.key_dim].transpose(0, 1), key))
            full_value = torch.cat(
                (
                    history[
                        ..., self.key_dim : self.key_dim + self.value_dim
                    ].transpose(0, 1),
                    value,
                )
            )
        output = prefill_attention(
            query,
            full_key,
            full_value,
            prefix,
            scale,
            chunk_size=self.config.prefill_chunk_size,
            indices=indices,
            logit_cap=logit_cap,
        )
        # prepare_append is now a no-op; the persistent summaries, active page,
        # and host history are identical to the ordinary committed-token path.
        self.append(key, value, prefix)
        return self._finish_attention(output)

    def materialize_history(self):
        if self.backup_done is not None:
            self.backup_done.synchronize()
        n = self.length
        if n == 0:
            return torch.empty(
                (self.heads, 0, self.storage_dim), dtype=self.dtype, device=self.device
            )
        if self.host is None:
            active = (n - 1) // self.config.block_size
            packed = torch.cat(
                [
                    self.block(i if i < active else self.config.history_blocks)
                    for i in range(active + 1)
                ],
                1,
            )[:, :n]
        else:
            packed = self.host.flatten(1, 2)[:, :n].to(self.device, non_blocking=True)
        return packed

    def dense_prefill(self, query, prefix, scale, *, indices=None, logit_cap=0):
        """Dense and sparse requests retain the same logical history order."""
        if (
            self.page_table
            and not self.sparse
            and self.pipeline.cuda
            and indices is None
        ):
            from sglang.kernels.ops.attention.extend_attention import (
                extend_attention_fwd_unified,
            )

            # Reuse SGLang's dense paged extend kernel directly. Logical token
            # indices address the same head-wise table as sparse decode; no
            # contiguous KV gather or duplicate dense cache is required.
            view = self.page_table.pool.slabs[0].permute(0, 2, 1, 3)
            locs = torch.arange(self.length, dtype=torch.int64, device=self.device)
            active = (self.length - 1) // self.config.block_size
            locs = torch.where(
                locs // self.config.block_size == active,
                self.config.history_blocks * self.config.block_size
                + locs % self.config.block_size,
                locs,
            )
            output = query.new_empty((query.shape[0], query.shape[1], self.value_dim))
            extend_attention_fwd_unified(
                query,
                output,
                view[..., : self.key_dim],
                view[..., self.key_dim : self.key_dim + self.value_dim],
                1.0,
                1.0,
                torch.tensor(
                    [0, query.shape[0]], dtype=torch.int32, device=self.device
                ),
                torch.tensor([0, self.length], dtype=torch.int32, device=self.device),
                locs,
                torch.tensor([prefix], dtype=torch.int32, device=self.device),
                query.shape[0],
                sm_scale=scale,
                logit_cap=logit_cap,
                page_size=self.config.block_size,
                kv_page_table=self.page_table.pointers,
                packed_value_offset=self.key_dim,
            )
            return self._finish_attention(output)
        packed = self.materialize_history()
        return self._finish_attention(
            prefill_attention(
                query,
                packed[..., : self.key_dim].transpose(0, 1),
                packed[..., self.key_dim : self.key_dim + self.value_dim].transpose(
                    0, 1
                ),
                prefix,
                scale,
                chunk_size=self.config.prefill_chunk_size,
                indices=indices,
                logit_cap=logit_cap,
            )
        )


class OasisKVCache:
    """Request generation ownership shared by all target layers and backends."""

    def __init__(self, config: OasisKVConfig, device, max_requests=None):
        self.config = config
        self.pipeline = OasisKVPipeline(device)
        self.layers = {}
        self.page_pools = {}
        self.max_requests = max_requests

    def new_layer(self, layer_id, heads, kd, vd, dtype, device):
        geometry = (layer_id, heads, kd, vd, dtype)
        if geometry not in self.page_pools:
            storage = ((kd + vd) * dtype.itemsize + 15) // 16 * 16 // dtype.itemsize
            self.page_pools[geometry] = OasisKVPagePool(
                heads,
                self.config.block_size,
                storage,
                dtype,
                device,
                (
                    None
                    if self.max_requests is None
                    else self.max_requests * (self.config.history_blocks + 1)
                ),
            )
        return OasisKVLayerCache(
            self.config,
            heads,
            kd,
            vd,
            dtype,
            device,
            self.pipeline,
            self.page_pools[geometry],
        )

    def get(self, slot, generation, layer_id, key, value):
        identity = (slot, generation, layer_id)
        if identity not in self.layers:
            for old in list(self.layers):
                if old[0] == slot and old[1] != generation:
                    self.layers.pop(old).release()
            self.layers[identity] = self.new_layer(
                layer_id,
                key.shape[1],
                key.shape[2],
                value.shape[2],
                key.dtype,
                key.device,
            )
        return self.layers[identity]

    def release(self, slot):
        for identity in list(self.layers):
            if identity[0] == slot:
                self.layers.pop(identity).release()

    def clear(self):
        for layer in self.layers.values():
            layer.release()
        self.layers.clear()
