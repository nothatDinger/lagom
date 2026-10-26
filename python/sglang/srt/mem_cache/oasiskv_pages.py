"""Shared physical KV pages for dense and sparse OasisKV requests.

Slabs never move: growing the pool does not invalidate captured addresses.
Request page tables are fixed-size and change only outside replay, after the
request's readers/writers finish. Head-wise logical history is a separate map.
"""

import torch


class OasisKVPagePool:
    def __init__(self, heads, block, storage, dtype, device, max_pages=None):
        self.shape = (heads, block, storage)
        self.dtype, self.device = dtype, device
        self.max_pages = max_pages
        self.slabs = []
        self.pages = []
        self.free_ids = []
        self.allocated = set()

    @property
    def nbytes(self):
        return sum(s.numel() * s.element_size() for s in self.slabs)

    def allocate(self):
        if not self.free_ids:
            count = (
                16
                if self.max_pages is None
                else min(16, self.max_pages - len(self.pages))
            )
            if count <= 0:
                raise RuntimeError("OasisKV physical page pool exhausted")
            slab = torch.empty(
                (count, *self.shape), dtype=self.dtype, device=self.device
            )
            start = len(self.pages)
            self.slabs.append(slab)
            self.pages.extend(slab.unbind())
            self.free_ids.extend(range(start + count - 1, start - 1, -1))
        index = self.free_ids.pop()
        self.allocated.add(index)
        self.pages[index].zero_()
        return index

    def release(self, pages):
        for page in set(pages):
            if page not in self.allocated:
                raise RuntimeError("OasisKV page double free")
            self.allocated.remove(page)
            self.free_ids.append(page)


class OasisKVPageTable:
    def __init__(self, pool, slots):
        self.pool, self.slots = pool, slots
        self.ids = [-1] * (slots + 1)
        self.active = None
        self.logical_to_physical = torch.full(
            (slots + 1,), -1, dtype=torch.int64, device=pool.device
        )
        self.pointers = torch.zeros(
            (pool.shape[0], slots + 1), dtype=torch.int64, device=pool.device
        )

    def ensure(self, length, *, sparse):
        if not length:
            return
        block = self.pool.shape[1]
        active = max(0, (length - 1) // block)
        # Active has a fixed logical slot; dense completed pages use identity
        # slots. At transition these same physical pages become sparse slots.
        history = self.slots if sparse else min(active, self.slots)
        changed = False
        if (
            self.active is not None
            and active > self.active
            and self.active < self.slots
            and self.ids[self.active] < 0
        ):
            self.ids[self.active], self.ids[-1] = self.ids[-1], -1
            changed = True
        for slot in range(history):
            if self.ids[slot] < 0:
                self.ids[slot] = self.pool.allocate()
                changed = True
        if self.ids[-1] < 0:
            self.ids[-1] = self.pool.allocate()
            changed = True
        if changed:
            self.refresh()
        self.active = active

    def needs_update(self, length, *, sparse):
        """Whether ensure() will write the table read by asynchronous kernels."""
        if not length:
            return False
        active = (length - 1) // self.pool.shape[1]
        history = self.slots if sparse else min(active, self.slots)
        return self.ids[-1] < 0 or any(page < 0 for page in self.ids[:history])

    def refresh(self):
        self.logical_to_physical.copy_(
            torch.tensor(self.ids, dtype=torch.int64, device=self.pool.device)
        )
        addresses = [
            [
                0 if page < 0 else self.pool.pages[page][h].data_ptr()
                for page in self.ids
            ]
            for h in range(self.pool.shape[0])
        ]
        self.pointers.copy_(
            torch.tensor(addresses, dtype=torch.int64, device=self.pool.device)
        )

    def block(self, slot):
        if self.ids[slot] < 0:
            raise RuntimeError("OasisKV logical page is not allocated")
        return self.pool.pages[self.ids[slot]]

    def materialize(self):
        zero = torch.zeros(
            self.pool.shape, dtype=self.pool.dtype, device=self.pool.device
        )
        return torch.stack(
            [zero if page < 0 else self.pool.pages[page] for page in self.ids], 1
        )

    def release(self):
        self.pool.release(page for page in self.ids if page >= 0)
        self.ids = [-1] * (self.slots + 1)
