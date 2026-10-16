"""PR #40670 pinned layout formulas, separated from serving configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Optional

MAX_MULTI_STEP_TOPK_NUM = 8192


class HiSparseKVTransferKind(str, Enum):
    LINEAR = "linear"
    DSV4_PAGED = "dsv4_paged"


@dataclass(frozen=True)
class HiSparseSpecLayout:
    """Logical layout for speculative HiSparse.

    0                    H-1 H          H+1               H+1+E                H+T
    ┌───────────────────────┬───────────┬─────────────────┬─────────────────────┐
    │ hot buffer: H         │ canonical │ extra: E        |scratch: S           │
    │ swap in/out           │ C=1       │ draft token kv  │mult step swap buffer│
    └───────────────────────┴───────────┴─────────────────┴─────────────────────┘
    H: device_buffer_size
    T: page_size
    """

    canonical_slots: ClassVar[int] = 1

    hot_slots: int
    tail_capacity_slots: int
    compress_ratio: int
    verify_width: int
    top_k: int

    @property
    def speculative_slots(self) -> int:
        return (self.verify_width + self.compress_ratio - 1) // self.compress_ratio

    @property
    def scratch_slots(self) -> int:
        return self.tail_capacity_slots - self.canonical_slots - self.speculative_slots

    @property
    def canonical_offset(self) -> int:
        return self.hot_slots

    @property
    def speculative_offset(self) -> int:
        return self.canonical_offset + self.canonical_slots

    @property
    def scratch_offset(self) -> int:
        return self.speculative_offset + self.speculative_slots

    @property
    def active_tail_slots(self) -> int:
        return self.canonical_slots + self.speculative_slots

    @property
    def active_buffer_slots(self) -> int:
        return self.hot_slots + self.active_tail_slots

    @property
    def total_buffer_slots(self) -> int:
        return self.hot_slots + self.tail_capacity_slots

    @property
    def occurrences(self) -> int:
        return self.verify_width * self.top_k

    @property
    def hash_slots(self) -> int:
        return 1 << (2 * self.hot_slots - 1).bit_length()

    def policy_width(self, num_req_slots: int) -> int:
        return max(num_req_slots, self.hot_slots)

    def state_width(self, num_req_slots: int) -> int:
        width = max(4 * num_req_slots, 5 * self.occurrences)
        # The CUDA workspace reinterprets pairs of int32 values as uint64.
        return width + width % 2

    def fixed_state_bytes(self, *, num_layers: int, num_req_slots: int) -> int:
        """Exact bytes allocated by HiSparseCoordinator for speculative state."""

        int32_bytes = 4
        int64_bytes = 8
        cache_index = num_layers * num_req_slots * 2 * self.hash_slots * int64_bytes
        cache_policy = (
            num_layers
            * (num_req_slots + 1)
            * self.policy_width(num_req_slots)
            * int32_bytes
        )
        scratch_state = (
            num_layers
            * (num_req_slots + 1)
            * self.state_width(num_req_slots)
            * int32_bytes
        )
        verify_buffers = (
            2 * num_req_slots * self.verify_width * self.top_k * int32_bytes
        )
        finalize_snapshots = num_req_slots * (int64_bytes + int64_bytes + int32_bytes)
        return (
            cache_index
            + cache_policy
            + scratch_state
            + verify_buffers
            + finalize_snapshots
        )


@dataclass(frozen=True)
class HiSparseSpecPlan:
    layout: HiSparseSpecLayout
    transfer_kind: HiSparseKVTransferKind


def build_hisparse_spec_layout(
    *,
    hot_slots: int,
    tail_capacity_slots: int,
    compress_ratio: int,
    verify_width: Optional[int],
    top_k: int,
) -> HiSparseSpecLayout:
    if verify_width is None:
        raise ValueError(
            "HiSparse speculative verify requires speculative_num_draft_tokens."
        )

    dimensions = {
        "hot_slots": hot_slots,
        "tail_capacity_slots": tail_capacity_slots,
        "compress_ratio": compress_ratio,
        "verify_width": verify_width,
        "top_k": top_k,
    }
    invalid_types = [
        name
        for name, value in dimensions.items()
        if not isinstance(value, int) or isinstance(value, bool)
    ]
    if invalid_types:
        raise ValueError(
            "HiSparse speculative layout dimensions must be integers; invalid: "
            + ", ".join(f"{name}={dimensions[name]!r}" for name in invalid_types)
        )

    layout = HiSparseSpecLayout(
        hot_slots=hot_slots,
        tail_capacity_slots=tail_capacity_slots,
        compress_ratio=compress_ratio,
        verify_width=verify_width,
        top_k=top_k,
    )
    if not (
        layout.compress_ratio > 0
        and layout.verify_width > 0
        and layout.top_k > 0
        and layout.tail_capacity_slots > 0
        and layout.occurrences <= layout.hot_slots
        and layout.occurrences <= MAX_MULTI_STEP_TOPK_NUM
        and layout.speculative_slots > 0
        and layout.scratch_slots >= 0
    ):
        raise ValueError(
            "Invalid HiSparse speculative layout: "
            f"H={layout.hot_slots}, C={layout.canonical_slots}, "
            f"E={layout.speculative_slots}, S={layout.scratch_slots}, "
            f"T={layout.tail_capacity_slots}, W={layout.verify_width}, "
            f"K={layout.top_k}, R={layout.compress_ratio}."
        )
    return layout
