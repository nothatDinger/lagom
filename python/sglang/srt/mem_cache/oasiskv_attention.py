"""Static launch policy and graph-owned scratch for physical-page attention."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.triton_ops import oasiskv as kernels
from sglang.srt.layers.attention.triton_ops.oasiskv_attention import attention_grouped


def use_grouped_attention(config, dtype, group, kd, vd):
    # Tiny MHA and large MLA heads retain the SIMT path. Blindly applying the
    # native MLA kernel's 64/128-head geometry would waste work for Qwen GQA.
    return (
        config.attention_kernel != "legacy"
        and dtype in (torch.float16, torch.bfloat16)
        and group >= 2
        and 0 < kd <= 256
        and 0 < vd <= 256
        and config.block_size <= 64
        and config.block_size & (config.block_size - 1) == 0
    )


@dataclass(frozen=True)
class AttentionGeometry:
    head_tile: int
    splits: int
    tiles_per_split: int
    tile: int = 32
    warps: int = 4


def attention_geometry(
    batch, heads, group, width, slots, block, sm_count, kd=128, vd=128
):
    head_tile = min(8, triton.next_power_of_2(group))
    groups = batch * heads * triton.cdiv(group, head_tile)
    # Enough splits to occupy the GPU after grouping heads, while bounding the
    # reduction workspace and avoiding tiny fragments at high concurrency.
    splits = min(
        32,
        max(1, triton.cdiv(slots * block, 128)),
        triton.next_power_of_2(triton.cdiv(2 * sm_count, groups)),
    )
    tile = max(32, block)
    return AttentionGeometry(
        head_tile,
        splits,
        triton.cdiv(slots * block, splits * tile),
        tile,
        8 if max(kd, vd) > 128 else 4,
    )


class GroupedAttentionWorkspace:
    def __init__(self, batch, heads, group, kd, vd, config, device, *, max_splits=None):
        self.heads, self.group, self.kd, self.vd = heads, group, kd, vd
        self.slots, self.block = config.history_blocks, config.block_size
        self.pages = torch.empty(
            (batch, heads, self.slots + 1, 2), dtype=torch.int64, device=device
        )
        # Called at workspace construction, outside graph capture/replay.
        self.sm_count = (
            torch.cuda.get_device_properties(device).multi_processor_count
            if torch.device(device).type == "cuda"
            else 132
        )
        # Reuse one fixed buffer across all capture buckets. Large batches need
        # fewer splits; do not reserve max_batch * 32 splits unnecessarily.
        rows = (
            batch * 2 * max_splits
            if max_splits is not None
            else max(
                b
                * width
                * attention_geometry(
                    b,
                    heads,
                    group,
                    width,
                    self.slots,
                    self.block,
                    self.sm_count,
                    kd,
                    vd,
                ).splits
                for b in range(1, batch + 1)
                for width in (1, 2)
            )
        )
        self.partial = torch.empty(
            rows * heads * group * (vd + 2), dtype=torch.float32, device=device
        )

    def prepare(self, table, ids, selected, storage, dtype):
        kernels.prepare_attention_pages[(table.shape[0], self.heads)](
            table,
            ids,
            selected,
            self.pages,
            HEADS=self.heads,
            SLOTS=self.slots,
            BLOCK=self.block,
            STORAGE=storage,
            DTYPE={torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}[dtype],
            NP=triton.next_power_of_2(self.slots + 1),
        )

    def forward(
        self,
        query,
        key,
        value,
        output,
        *,
        batch,
        width,
        storage,
        scale,
        cap,
        geometry=None
    ):
        g = geometry or attention_geometry(
            batch,
            self.heads,
            self.group,
            width,
            self.slots,
            self.block,
            self.sm_count,
            self.kd,
            self.vd,
        )
        if (
            batch * width * self.heads * self.group * g.splits * (self.vd + 2)
            > self.partial.numel()
        ):
            raise ValueError("Attention split geometry exceeds the fixed workspace")
        compiled = attention_grouped[
            (batch, self.heads * triton.cdiv(self.group, g.head_tile), g.splits)
        ](
            self.pages,
            query,
            key,
            value,
            self.partial,
            HEADS=self.heads,
            GROUP=self.group,
            KD=self.kd,
            VD=self.vd,
            STORAGE=storage,
            SLOTS=self.slots,
            BLOCK=self.block,
            WIDTH=width,
            SCALE=scale,
            CAP=cap,
            SPLITS=g.splits,
            TILES_PER_SPLIT=g.tiles_per_split,
            HEAD_TILE=g.head_tile,
            M=max(16, triton.next_power_of_2(g.head_tile * width)),
            TILE=g.tile,
            DK=max(16, triton.next_power_of_2(self.kd)),
            DV=max(16, triton.next_power_of_2(self.vd)),
            num_warps=g.warps,
        )
        kernels.attention_reduce[(batch * width, self.heads * self.group)](
            self.partial,
            output,
            QHEADS=self.heads * self.group,
            VD=self.vd,
            SPLITS=g.splits,
            NS=triton.next_power_of_2(g.splits),
            DV=triton.next_power_of_2(self.vd),
        )
        return compiled
