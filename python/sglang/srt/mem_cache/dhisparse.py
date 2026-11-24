"""Storage geometry and load-adaptive ownership for Adaptive DHiSparse.

The control plane deliberately has no CUDA dependencies. Transfers return completion
objects with ``query``/``synchronize``; the same ownership rules are exercised on CPU
and used by the serving coordinator. Required pages are never lent to history.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import torch


def page_round(value: int, page_size: int) -> int:
    if value < 0 or page_size <= 0:
        raise ValueError("Invalid page geometry")
    return (value + page_size - 1) // page_size * page_size


@dataclass(frozen=True)
class DHiSparseConfig:
    enabled: bool = True
    admission_reserve_pages: int = 1
    refill_pages_per_step: int = 16
    resident_first: bool = True

    def __post_init__(self):
        for name in ("admission_reserve_pages", "refill_pages_per_step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.enabled) is not bool or type(self.resident_first) is not bool:
            raise ValueError("DHiSparse switches must be booleans")


@dataclass(frozen=True)
class VerificationLayout:
    num_requests: int
    max_verify_width: int
    top_k: int
    page_size: int
    layer_slots: tuple[int, ...]
    request_slots: int | None = None
    reserve_pages: int = 0

    def __post_init__(self):
        if (
            min(self.num_requests, self.max_verify_width, self.top_k, self.page_size)
            <= 0
        ):
            raise ValueError("Verification dimensions must be positive")
        if not self.layer_slots or min(self.layer_slots) < 0:
            raise ValueError("Each payload source needs an arena assignment")
        if self.request_slots is not None and self.request_slots < self.num_requests:
            raise ValueError("Request slot table is smaller than the active batch")
        if self.reserve_pages < 0:
            raise ValueError("Negative admission reserve")

    @property
    def slots(self):
        return self.num_requests if self.request_slots is None else self.request_slots

    @property
    def packed_row_bytes(self):
        # FlashMLA rounds packed C4 pages to 576-byte alignment.
        return page_round(self.page_size * 584, 576) // self.page_size

    @property
    def tail_rows(self):
        # One committed C4 frontier plus the C4 boundaries a verify window can complete.
        return page_round(1 + (self.max_verify_width + 3) // 4, self.page_size)

    @property
    def rows_per_request(self):
        return page_round(self.max_verify_width * self.top_k, self.page_size)

    @property
    def arenas(self):
        return max(self.layer_slots) + 1

    @property
    def rows(self):
        return self.arenas * self.num_requests * self.rows_per_request

    def offsets(self, owned_rows_per_layer: int):
        """Arena row offsets relative to each layer view's base address."""
        layers = len(self.layer_slots)
        return tuple(
            (layers - layer) * owned_rows_per_layer
            + slot * self.num_requests * self.rows_per_request
            for layer, slot in enumerate(self.layer_slots)
        )

    def metadata_bytes(self, hot: int, context_rows: int) -> int:
        n, layers = self.num_requests, len(self.layer_slots)
        occurrences = self.max_verify_width * self.top_k
        hashes = 1 << (2 * (hot + occurrences) - 1).bit_length()
        tail = self.tail_rows
        # Persistent request maps include PD transfer slots and the graph dummy.
        # Hashes, miss/hit plans, results and LRU and compaction scratch are
        # shared across sequential sources and indexed by active batch row.
        return (
            self.slots * (context_rows * 12 + (hot + tail) * 8 + self.top_k * 8)
            + layers * self.slots * (hot * 20 + tail * 8 + 8)
            + self.arenas * n * (hashes * 12 + occurrences * 28 + hot * 4 + 4)
        )

    def _reserved_rows(self, row_bytes, hot, context_rows):
        layers = len(self.layer_slots)
        extra = self.rows * row_bytes + self.metadata_bytes(hot, context_rows)
        return page_round(
            (extra + layers * row_bytes - 1) // (layers * row_bytes), self.page_size
        )

    def _required_rows(self, hot):
        return (
            self.num_requests * (page_round(hot, self.page_size) + self.tail_rows)
            + self.reserve_pages * self.page_size
        )

    def fitting_requests(self, budget_rows, row_bytes, hot, context_rows):
        """Largest automatic concurrency whose arenas, metadata, hot, tail and reserve fit."""
        lo, hi = 0, self.num_requests
        while lo < hi:
            n = (lo + hi + 1) // 2
            candidate = replace(
                self, num_requests=n, request_slots=n + self.slots - self.num_requests
            )
            minimum = candidate._reserved_rows(
                row_bytes, hot, context_rows
            ) + candidate._required_rows(hot)
            if minimum <= budget_rows:
                lo = n
            else:
                hi = n - 1
        return lo

    def owned_rows(
        self, budget_rows: int, row_bytes: int, hot: int, context_rows: int
    ) -> int:
        removed = self._reserved_rows(row_bytes, hot, context_rows)
        result = budget_rows // self.page_size * self.page_size - removed
        required = self._required_rows(hot)
        if result < required:
            raise ValueError(
                "DHiSparse required hot/tail/staging/metadata exceed the KV budget: "
                f"payload rows after reservation={result}, required={required}. "
                "Reduce max-running-requests, hot cache or verification width."
            )
        return result


def allocate_payload_slab(
    *,
    owned_units: int,
    unit_shape: tuple[int, ...],
    dtype,
    device,
    layout: VerificationLayout,
    rows_per_unit: int = 1,
):
    """Contiguous layer views sharing the same physical staging allocation.

    Ordinary row indices stay layer-local. Each view extends beyond its owned
    payload through the arenas; only resolver-produced indices address that part.
    ``rows_per_unit`` is the packed page size for DSv4 and one for linear MLA.
    """
    if owned_units <= 0 or layout.rows % rows_per_unit:
        raise ValueError("Payload slab must be page aligned")
    layers = len(layout.layer_slots)
    slab = torch.zeros(
        (layers * owned_units + layout.rows // rows_per_unit, *unit_shape),
        dtype=dtype,
        device=device,
    )
    views = [slab[i * owned_units :] for i in range(layers)]
    return slab, views, layout.offsets(owned_units * rows_per_unit)


@dataclass
class OptionalHistory:
    request: int
    imported_rows: int
    pages: Any
    backed: bool
    populate_cursor: int = 0
    ready_rows: int = 0
    pending: list[tuple[Any, Any, int]] = field(default_factory=list)


class ResidencyManager:
    """Whole-request reclamation and bounded round-robin refill (paper §3.1).

    Pages are opaque to this class. All callbacks execute on the scheduler thread.
    ``retire`` must wait for earlier readers before freeing; ``backup`` must make
    every retained page recoverable before returning. ``publish`` is called only
    after the corresponding copy event completes.
    """

    def __init__(
        self,
        *,
        page_size: int,
        config: DHiSparseConfig,
        available: Callable[[], int],
        allocate: Callable,
        copy: Callable,
        publish: Callable,
        unpublish: Callable,
        backup: Callable,
        retire: Callable,
    ):
        self.page_size, self.config = page_size, config
        self.available, self.allocate, self.copy = available, allocate, copy
        self.publish, self.unpublish = publish, unpublish
        self.backup, self.retire = backup, retire
        self.requests: OrderedDict[int, OptionalHistory] = OrderedDict()
        self.round_robin: deque[int] = deque()

    def register(self, request: int, imported_rows: int, pages, *, backed: bool):
        if request in self.requests:
            raise ValueError(f"Request slot {request} is already owned")
        if len(pages) % self.page_size:
            raise ValueError("Optional history owns complete pages only")
        if len(pages) > imported_rows // self.page_size * self.page_size:
            raise ValueError("Optional pages exceed imported history")
        state = OptionalHistory(
            request,
            imported_rows,
            pages,
            backed,
            populate_cursor=len(pages),
            ready_rows=len(pages),
        )
        self.requests[request] = state
        self.round_robin.append(request)
        if len(pages):
            self.publish(request, 0, pages)
        return state

    @property
    def reclaimable_rows(self):
        return sum(
            len(s.pages) + sum(len(p) for _, p, _ in s.pending)
            for s in self.requests.values()
        )

    def poll(self):
        for state in self.requests.values():
            remaining = []
            for event, pages, start in state.pending:
                if event.query():
                    self.publish(state.request, start, pages)
                    state.pages = torch.cat((state.pages, pages))
                    state.ready_rows += len(pages)
                else:
                    remaining.append((event, pages, start))
            state.pending = remaining

    def _drain(self, state):
        for event, pages, start in state.pending:
            event.synchronize()
            self.publish(state.request, start, pages)
            state.pages = torch.cat((state.pages, pages))
            state.ready_rows += len(pages)
        state.pending.clear()

    def reclaim(self, required_rows: int) -> bool:
        if required_rows < 0:
            raise ValueError("Negative required allocation")
        if self.available() + self.reclaimable_rows < required_rows:
            return False
        for state in self.requests.values():
            if self.available() >= required_rows:
                return True
            if not len(state.pages) and not state.pending:
                continue
            self._drain(state)
            if not state.backed:
                self.backup(state)
                state.backed = True
            # Unpublish before retiring; retire joins all earlier readers.
            self.unpublish(state.request)
            self.retire(state.pages)
            state.pages = state.pages[:0]
            state.ready_rows = state.populate_cursor = 0
        return self.available() >= required_rows

    def refill(self):
        self.poll()
        budget = self.config.refill_pages_per_step * self.page_size
        reserve = self.config.admission_reserve_pages * self.page_size
        for _ in range(len(self.round_robin)):
            if budget <= 0:
                break
            request = self.round_robin.popleft()
            self.round_robin.append(request)
            state = self.requests[request]
            if not state.backed or state.pending:
                continue
            remaining = (
                state.imported_rows // self.page_size * self.page_size
                - state.populate_cursor
            )
            count = min(remaining, budget, max(0, self.available() - reserve))
            count = count // self.page_size * self.page_size
            if not count:
                continue
            if self._start_copy(state, count):
                budget -= count

    def populate_on_admission(self, request):
        """A PD import already has host backing; retain its full pages when they fit."""
        if not self.config.resident_first:
            return
        state = self.requests[request]
        complete = state.imported_rows // self.page_size * self.page_size
        free = self.available() - self.config.admission_reserve_pages * self.page_size
        if state.backed and not state.populate_cursor and 0 < complete <= free:
            self._start_copy(state, complete)

    def _start_copy(self, state, count):
        pages = self.allocate(count)
        if pages is None:
            return False
        start = state.populate_cursor
        try:
            event = self.copy(state.request, start, pages)
        except BaseException:
            self.retire(pages)
            raise
        state.pending.append((event, pages, start))
        state.populate_cursor += count
        return True

    def release(self, request: int):
        state = self.requests.get(request)
        if state is None:
            return
        self._drain(state)
        self.unpublish(request)
        self.retire(state.pages)
        del self.requests[request]
        self.round_robin.remove(request)


class RequiredPageAllocator:
    """Required allocations may reclaim optional pages, including at admission."""

    def __init__(self, allocator, residency: ResidencyManager):
        self.allocator, self.residency = allocator, residency

    def __getattr__(self, name):
        return getattr(self.allocator, name)

    def available_size(self):
        return max(
            0,
            self.allocator.available_size()
            + self.residency.reclaimable_rows
            - self.reserve,
        )

    @property
    def reserve(self):
        return self.residency.config.admission_reserve_pages * self.residency.page_size

    def alloc(self, size):
        if not self.residency.reclaim(size + self.reserve):
            return None
        return self.allocator.alloc(size)

    def alloc_extend(self, *args, **kwargs):
        # Required new pages are bounded from CPU sequence lengths. Avoid an
        # optimistic free-space report followed by a failing underlying allocator.
        prefix_cpu = args[1] if len(args) > 1 else kwargs["prefix_lens_cpu"]
        lengths_cpu = args[3] if len(args) > 3 else kwargs["seq_lens_cpu"]
        p = self.allocator.page_size
        rows = (
            int((((lengths_cpu + p - 1) // p) - ((prefix_cpu + p - 1) // p)).sum()) * p
        )
        if not self.residency.reclaim(rows + self.reserve):
            return None
        return self.allocator.alloc_extend(*args, **kwargs)

    @property
    def is_not_in_free_group(self):
        return self.allocator.is_not_in_free_group

    @is_not_in_free_group.setter
    def is_not_in_free_group(self, value):
        self.allocator.is_not_in_free_group = value
