"""Offline-only physical KV copies for a wide exact shadow forward.

The persistent HiSparse cache keeps its configured capacity and native LRU.
Copies protect each query until attention finishes; they are not prefetch hits.
"""

import json
import os
from pathlib import Path

import torch
from sglang.kernels.ops.kvcache.hisparse import transfer_cache_dsv4_mla


def trace_resident_verify_width(runtime_width):
    path = os.environ.get("SGLANG_OASISKV_TRACE_CONFIG")
    if not path:
        return runtime_width
    config = json.loads(Path(path).read_text())
    if config.get("probe_workspace") != "row-copy":
        return runtime_width
    v = config.get("verify_candidates")
    if type(v) is not int or not 1 <= v < runtime_width:
        raise ValueError("Trace real verify width must fit the runtime proposal")
    return v + 1


class ShadowKVWorkspace:
    def __init__(self, coordinator, width):
        self.coordinator = coordinator
        self.width = width
        self.allocator = coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator
        self.count = width * coordinator.top_k

    def __enter__(self):
        a = self.allocator
        self.saved_free, self.saved_release = a.free_pages, a.release_pages
        size = (self.count + a.page_size - 1) // a.page_size * a.page_size
        try:
            self.locations = a.alloc(size)
            if self.locations is None:
                raise RuntimeError(
                    f"Exact shadow needs {size} temporary C4 slots outside the resident cache; "
                    "increase the global KV pool/context-length, not resident_units"
                )
            self.expected_free, self.expected_release = a.free_pages, a.release_pages
            self.rows = self.locations[: self.count].view(
                self.width, self.coordinator.top_k
            )
            return self
        except Exception:
            a.free_pages, a.release_pages = self.saved_free, self.saved_release
            raise

    def copy_row(self, layer, row, raw, selected, history_units):
        valid = (raw >= 0) & (raw < history_units) & (selected >= 0)
        source = selected[valid].to(torch.int64)
        destination = self.rows[row][valid.reshape(-1)]
        if source.numel():
            ptrs = self.coordinator.mem_pool_host.device_ptrs[layer : layer + 1]
            transfer_cache_dsv4_mla(ptrs, ptrs, source, destination)
            selected[valid] = destination.to(selected.dtype)
        return raw[valid].cpu().tolist(), destination.cpu().tolist()

    def __exit__(self, exc_type, exc, tb):
        # This experiment disables scheduling overlap. run_compact uses already
        # reserved logical/scratch indices and must not allocate C4 pages.
        torch.cuda.synchronize()
        a = self.allocator
        if (
            a.free_pages is not self.expected_free
            or a.release_pages is not self.expected_release
        ):
            raise RuntimeError("Unexpected C4 allocation during isolated exact shadow")
        # Restore the precise free-list order, not just its size. No workspace
        # pages survive the forward or compete with the next request's prefill.
        a.free_pages, a.release_pages = self.saved_free, self.saved_release
