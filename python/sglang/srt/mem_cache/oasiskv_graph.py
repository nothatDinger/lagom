"""Batched, graph-recordable decode for the generic OasisKV cache.

Host allocation and request ownership stay outside replay. A fixed GPU table
indirects into each live request's storage; all token-dependent work, including
UVA backup/refill and per-layer prefetch, is performed by captured kernels.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.srt.mem_cache.oasiskv_async import LayerPrefetchGraphs, PrefetchGraphBatch
from sglang.srt.mem_cache.oasiskv_attention import (
    GroupedAttentionWorkspace,
    use_grouped_attention,
)
from sglang.srt.mem_cache.oasiskv_graph_inputs import GraphInputs


def copy_launch_geometry(
    *, batch, heads, slots, block, storage, element_size, sm_count, ctas_per_sm
):
    """Static capture-safe geometry; no readback of per-step miss counts.

    2 KiB tiles issue one aligned 16-byte vector per thread with four warps.
    Cap the entire grid, including at large batches. CTAs iterate over groups
    when there are more request/head groups than the budget. This limits
    background concurrency without truncating its admission plan. It does not
    reserve SMs or guarantee a bound on shared memory-system interference.
    """
    tile = min(triton.next_power_of_2(block * storage), 2048 // element_size)
    tiles = triton.cdiv(block * storage, tile)
    budget = sm_count * ctas_per_sm
    groups = min(batch * heads, budget)
    workers = min(slots * tiles, max(1, budget // groups))
    return tile, (groups, workers)


class GraphPlan:
    def __init__(self, batch, heads, slots, device):
        shape = (batch, heads, slots)
        self.ids = torch.empty(shape, dtype=torch.int64, device=device)
        self.ages = torch.empty_like(self.ids)
        self.sources = torch.empty_like(self.ids)
        self.destinations = torch.empty_like(self.ids)
        self.counts = torch.empty((batch, heads), dtype=torch.int32, device=device)
        self.selected = torch.empty(shape, dtype=torch.bool, device=device)
        # Query hardware before graph capture, not during replay or a fork.
        self.sm_count = (
            torch.cuda.get_device_properties(device).multi_processor_count
            if torch.device(device).type == "cuda"
            else 1
        )

    def select(self, table, candidates, *, cap, background, use_refill_base=False):
        batch = table.shape[0]
        heads, slots = self.ids.shape[1:]
        kernels.plan_delta[(batch, heads)](
            table,
            candidates,
            self.ids,
            self.ages,
            self.sources,
            self.destinations,
            self.counts,
            self.selected,
            HEADS=heads,
            SLOTS=slots,
            CAP=slots if cap is None else min(cap, slots),
            BACKGROUND=background,
            N=triton.next_power_of_2(slots),
            SEARCH_STEPS=triton.next_power_of_2(slots).bit_length(),
            USE_REFILL_BASE=use_refill_base,
        )

    def transfer(
        self,
        table,
        config,
        storage,
        dtype,
        *,
        background,
        next_plan=None,
        attention_pages=None,
    ):
        heads, slots = self.ids.shape[1:]
        batch = table.shape[0]
        if config.copy_kernel == "legacy":
            kernel, tile, workers = kernels.copy_misses_legacy, 1024, 4
            grid, extra = (batch, heads, workers), {}
        else:
            kernel = kernels.copy_misses
            tile, grid = copy_launch_geometry(
                batch=batch,
                heads=heads,
                slots=slots,
                block=config.block_size,
                storage=storage,
                element_size=dtype.itemsize,
                sm_count=self.sm_count,
                ctas_per_sm=(
                    config.prefetch_ctas_per_sm
                    if background
                    else config.refill_ctas_per_sm
                ),
            )
            workers, extra = grid[1], {"GROUPS": batch * heads}
        kernel[grid](
            table,
            self.sources,
            self.destinations,
            self.counts,
            HEADS=heads,
            SLOTS=slots,
            BLOCK=config.block_size,
            STORAGE=storage,
            DTYPE={
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }[dtype],
            TILE=tile,
            WORKERS=workers,
            num_warps=4,
            **extra,
        )
        # Publish only after all block-copy CTAs finish. On the side stream this
        # also follows attention, so neither bytes nor residency race its reads.
        kernels.publish_plan[(batch, heads)](
            table,
            self.ids,
            self.ages,
            HEADS=heads,
            SLOTS=slots,
            N=triton.next_power_of_2(slots),
            NextIds=None if next_plan is None else next_plan.ids,
            NextAges=None if next_plan is None else next_plan.ages,
            SEED_NEXT=next_plan is not None,
            AttentionPages=attention_pages,
            Selected=self.selected,
            BLOCK=config.block_size,
            STORAGE=storage,
            DTYPE={
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float32: tl.float32,
            }[dtype],
        )


class GraphWorkspace:
    """Foreground scratch shared across layers with the same head geometry."""

    def __init__(self, batch, heads, group, kd, vd, config, max_blocks, device, dtype):
        slots = config.history_blocks
        self.real = GraphPlan(batch, heads, slots, device)
        self.candidates = torch.empty(
            (batch, heads, slots), dtype=torch.int64, device=device
        )
        self.scores = (
            None
            if config.native_sparse
            else torch.empty(
                (batch, 2, heads, max_blocks), dtype=torch.float32, device=device
            )
        )
        tokens = (slots + 1) * config.block_size + 1
        self.splits = min(8, triton.cdiv(tokens, 256))
        self.per_split = triton.cdiv(tokens, self.splits * 32) * 32
        self.partial = torch.empty(
            (batch * 2, heads * group, self.splits, vd + 2),
            dtype=torch.float32,
            device=device,
        )
        self.grouped = (
            GroupedAttentionWorkspace(batch, heads, group, kd, vd, config, device)
            if use_grouped_attention(config, dtype, group, kd, vd)
            else None
        )


@dataclass
class GraphLayer:
    heads: int
    group: int
    key_dim: int
    value_dim: int
    storage_dim: int
    dtype: torch.dtype
    draft: torch.Tensor
    plan: GraphPlan
    workspace: GraphWorkspace
    background_table: torch.Tensor
    background: object


class OasisKVGraphState:
    def __init__(self, cache, max_bs, max_context, num_layers, start_layer=0):
        self.cache = cache
        self.config = cache.config
        self.device = cache.pipeline.device
        self.max_bs = max_bs
        self.max_context = max_context
        self.max_blocks = triton.cdiv(max_context, self.config.block_size)
        self.start_layer = start_layer
        self.tables = torch.zeros(
            (num_layers, max_bs, kernels.FIELDS), dtype=torch.int64, device=self.device
        )
        self.tables[..., kernels.PREFIX] = -1
        self.layers = {}
        self.workspaces = {}
        self.pending = []
        self.cuda = self.device.type == "cuda"
        self.prepare_stream = (
            torch.cuda.Stream(device=self.device) if self.cuda else None
        )
        self.replay_batch_size = max_bs
        self.completed = torch.cuda.Event() if self.cuda else None
        self.inputs = GraphInputs(self)
        self.prefetch_batch = PrefetchGraphBatch()

    def get_layer(self, layer_id, query, key, value):
        if layer_id not in self.layers:
            for request in self.inputs.requests.values():
                request.dirty = True
            heads, kd, vd = key.shape[1], key.shape[2], value.shape[2]
            group = query.shape[1] // heads
            geometry = (heads, group, kd, vd, key.dtype)
            if geometry not in self.workspaces:
                self.workspaces[geometry] = GraphWorkspace(
                    self.max_bs,
                    heads,
                    group,
                    kd,
                    vd,
                    self.config,
                    self.max_blocks,
                    self.device,
                    key.dtype,
                )
            storage = (
                triton.cdiv((kd + vd) * key.element_size(), 16)
                * 16
                // key.element_size()
            )
            self.layers[layer_id] = GraphLayer(
                heads,
                group,
                kd,
                vd,
                storage,
                key.dtype,
                torch.empty(
                    (self.max_bs, heads, self.config.history_blocks),
                    dtype=torch.int64,
                    device=self.device,
                ),
                GraphPlan(self.max_bs, heads, self.config.history_blocks, self.device),
                self.workspaces[geometry],
                torch.zeros(
                    (self.max_bs, kernels.FIELDS), dtype=torch.int64, device=self.device
                ),
                (
                    LayerPrefetchGraphs(
                        self.cache.pipeline.select, self.cache.pipeline.transfer
                    )
                    if self.cuda
                    else None
                ),
            )
            self.layers[layer_id].background_table[:, kernels.PREFIX] = -1
        layer = self.layers[layer_id]
        if (key.shape[1:], value.shape[1:], query.shape[1:], key.dtype) != (
            (layer.heads, layer.key_dim),
            (layer.heads, layer.value_dim),
            (layer.heads * layer.group, layer.key_dim),
            layer.dtype,
        ):
            raise ValueError("OasisKV graph head geometry changed after capture")
        return layer

    def prepare(self, slots, generations, prefixes, *, paired, graph_batch_size=None):
        self.inputs.prepare(
            slots, generations, prefixes, paired, graph_batch_size or self.max_bs
        )

    def finish_in_graph(self):
        # Called by warmup first; during model capture this is a cache hit.
        # Side work remains independent of the foreground capture.
        if self.cuda:
            self.prefetch_batch.capture(
                self._capture_batch_size,
                [layout.background for layout in self.layers.values()],
            )

    def finish_replay(self):
        """Outside graph: reflect the enqueued committed root in host ownership."""
        if self.completed is not None:
            self.completed.record(torch.cuda.current_stream(self.device))
            if self.replay_paired:
                if self.replay_batch_size in self.prefetch_batch.graphs:
                    self.prefetch_batch.replay(self.replay_batch_size)
                else:
                    # Direct kernel/eager callers may omit the model epilogue.
                    for layout in self.layers.values():
                        layout.background.replay(self.replay_batch_size)
        paired = self.replay_paired
        for request in self.pending:
            request.length += 1
            request.step_delta += int(paired)
            for state in request.states:
                state.length = request.length
                state.step += int(paired)
                state.prefetch_submissions += int(paired)
                state.initialized = True
                state.backup_done = state.refill_done = self.completed
                state.read_done = self.completed
                background = getattr(state, "graph_background", None)
                state.pending = background.done if paired and background else None
                state.selection_done = (
                    background.selection_done if paired and background else None
                )
        self.pending = []

    def forward(self, query, key, value, layer, batch_size, *, paired, topk=None):
        layout = self.get_layer(layer.layer_id, query, key, value)
        self._capture_batch_size = batch_size
        if layer.layer_id == self.start_layer:
            self.inputs.materialize(batch_size)
        if layout.background is not None:
            layout.background.consume()
        table = self.tables[layer.layer_id - self.start_layer, :batch_size]
        w = layout.workspace
        heads, kd, vd, storage = (
            layout.heads,
            layout.key_dim,
            layout.value_dim,
            layout.storage_dim,
        )
        slots, block = self.config.history_blocks, self.config.block_size
        width = 2 if paired else 1
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        kernels.append_token[(batch_size, heads)](
            table,
            key,
            value,
            HEADS=heads,
            KD=kd,
            VD=vd,
            STORAGE=storage,
            SLOTS=slots,
            BLOCK=block,
            WIDTH=width,
            NATIVE=self.config.native_sparse,
            D=triton.next_power_of_2(storage),
        )
        if self.config.native_sparse:
            kernels.native_candidates[(batch_size, heads)](
                table,
                topk.contiguous(),
                w.candidates,
                layout.draft,
                HEADS=heads,
                SLOTS=slots,
                TOPK=topk.shape[-1],
                WIDTH=width,
                N=triton.next_power_of_2(slots),
            )
        else:
            kernels.score_blocks[
                (batch_size * heads, width, triton.cdiv(self.max_blocks, 16))
            ](
                table,
                query,
                w.scores,
                HEADS=heads,
                GROUP=layout.group,
                KD=kd,
                BLOCK=block,
                MAX_BLOCKS=self.max_blocks,
                WIDTH=width,
                D=triton.next_power_of_2(kd),
                TILE=16,
            )
            count = min(slots, self.max_blocks)
            scores, ids = w.scores[:batch_size, :width].topk(count, dim=-1)
            ids.masked_fill_(~scores.isfinite(), -1)
            w.candidates[:batch_size].fill_(-1)
            w.candidates[:batch_size, :, :count].copy_(ids[:, 0])
            if paired:
                layout.draft[:batch_size].fill_(-1)
                layout.draft[:batch_size, :, :count].copy_(ids[:, 1])
        w.real.select(table, w.candidates, cap=None, background=False)
        w.real.transfer(
            table,
            self.config,
            storage,
            key.dtype,
            background=False,
            next_plan=layout.plan if paired else None,
            attention_pages=w.grouped.pages if w.grouped is not None else None,
        )

        if paired:
            snapshot = layout.background_table[:batch_size]
            if batch_size not in layout.background.graphs:
                layout.background.synchronize()
                # Startup or a new capture bucket must warm only inactive rows,
                # including after an earlier live replay populated the snapshot.
                snapshot[:, kernels.PREFIX].fill_(-1)

            def select():
                layout.plan.select(
                    snapshot,
                    layout.draft,
                    cap=self.config.max_new_blocks,
                    background=True,
                    use_refill_base=True,
                )

            def transfer():
                layout.plan.transfer(
                    snapshot, self.config, storage, key.dtype, background=True
                )

            layout.background.capture(batch_size, select, transfer)
            snapshot.copy_(table)
            layout.background.prediction_ready.record()
        output = torch.empty(
            (batch_size * width, heads * layout.group, vd),
            dtype=query.dtype,
            device=self.device,
        )
        if w.grouped is not None and query.dtype == key.dtype == value.dtype:
            w.grouped.forward(
                query,
                key,
                value,
                output,
                batch=batch_size,
                width=width,
                storage=storage,
                scale=layer.scaling,
                cap=layer.logit_cap,
            )
            if paired:
                layout.background.read_done.record()
            return output.reshape(batch_size * width, -1)
        kernels.attention_partial[(batch_size, heads * layout.group, w.splits)](
            table,
            query,
            key,
            value,
            w.real.selected,
            w.partial,
            HEADS=heads,
            GROUP=layout.group,
            KD=kd,
            VD=vd,
            STORAGE=storage,
            SLOTS=slots,
            BLOCK=block,
            WIDTH=width,
            SCALE=layer.scaling,
            CAP=layer.logit_cap,
            SPLITS=w.splits,
            PER_SPLIT=w.per_split,
            DK=triton.next_power_of_2(kd),
            DV=triton.next_power_of_2(vd),
            TILE=32,
            num_warps=8 if max(kd, vd) > 128 else 4,
        )
        kernels.attention_reduce[(batch_size * width, heads * layout.group)](
            w.partial,
            output,
            QHEADS=heads * layout.group,
            VD=vd,
            SPLITS=w.splits,
            NS=triton.next_power_of_2(w.splits),
            DV=triton.next_power_of_2(vd),
        )
        if paired:
            layout.background.read_done.record()
        return output.reshape(batch_size * width, -1)
