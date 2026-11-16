"""CPU regressions for logical versus physical C4 prefill locations.

Load only the production method to avoid importing CUDA backend dependencies.
"""
import ast
import copy
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py"


def load_ensure():
    cls = next(
        node for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "SparsePrefillChunkCache"
    )
    method = copy.deepcopy(next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_c4"
    ))
    tree = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        method,
    ], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), namespace)
    return namespace["ensure_c4"]


ENSURE = load_ensure()


def inputs():
    cache = SimpleNamespace(
        c4_flat_token_ids=None, num_reqs=2, max_seq_len=20,
        seq_lens=torch.tensor([20, 4], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        swa_offsets=torch.tensor([0, 3, 7], dtype=torch.int32),
    )
    # Both query rows of request 0 share a logical page table. Request 1
    # has only one live C4 row; its unused page contains a stale large ID.
    pages = torch.tensor([[8, 11], [8, 11], [2, 1 << 30]], dtype=torch.int64)
    mapping = torch.zeros(64, dtype=torch.int64)
    mapping[32:36] = torch.arange(4, 8)
    mapping[44] = 8
    mapping[8] = 9
    return cache, pages, mapping


@pytest.mark.parametrize("proxy", [False, True])
def test_hisparse_translates_logical_pages_and_ignores_stale_request_tail(proxy):
    cache, pages, mapping = inputs()
    # Logical page 8 cannot address a three-page physical payload directly.
    assert pages[0, 0] * 4 >= 3 * 4
    ENSURE(
        cache, pages, 4,
        logical_to_device=weakref.proxy(mapping) if proxy else mapping,
        physical_num_pages=3,
    )
    assert cache.c4_flat_token_ids.dtype == torch.int32
    assert cache.c4_flat_token_ids.tolist() == [4, 5, 6, 7, 8, 9, 0, 0, 0, 0]
    assert cache.c4_compressed_base.tolist() == [0, 5]
    assert cache.c4_swa_base.tolist() == [10, 13]


def test_ordinary_pool_keeps_identity_addresses_for_live_rows():
    cache, _, _ = inputs()
    pages = torch.tensor([[2, 3], [2, 3], [1, 1 << 30]], dtype=torch.int64)
    ENSURE(cache, pages, 4, physical_num_pages=4)
    assert cache.c4_flat_token_ids.tolist() == [8, 9, 10, 11, 12, 4, 0, 0, 0, 0]


def test_missing_live_mapping_is_not_silently_replaced_with_slot_zero():
    cache, pages, mapping = inputs()
    mapping[32] = 0
    with pytest.raises(RuntimeError, match="no resident HiSparse mapping"):
        ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=3)
    assert cache.c4_flat_token_ids is None


def test_logical_bound_is_checked_before_mapping_lookup():
    cache, pages, mapping = inputs()
    pages[0, 0] = 16  # First live logical ID 64 is outside mapping[0:64].
    with pytest.raises(RuntimeError, match="logical ID exceeds"):
        ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=3)
    assert cache.c4_flat_token_ids is None


@pytest.mark.parametrize("bad_location", [12, 10**9])
def test_actual_physical_extent_is_checked_in_token_units(bad_location):
    cache, pages, mapping = inputs()
    mapping[32] = bad_location
    with pytest.raises(RuntimeError, match="physical payload capacity"):
        ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=3)
    assert cache.c4_flat_token_ids is None


def test_zero_complete_c4_rows_do_not_lookup_uninitialized_page_ids():
    cache, pages, mapping = inputs()
    cache.seq_lens = torch.tensor([3, 1], dtype=torch.int32)
    cache.max_seq_len = 3
    pages.fill_(1 << 30)
    ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=1)
    assert cache.c4_flat_token_ids.tolist() == [0, 0]


def test_metadata_extent_check_is_retained():
    cache, pages, mapping = inputs()
    cache.max_seq_len = 36  # Nine rows need three four-token pages.
    with pytest.raises(AssertionError, match="metadata capacity"):
        ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=3)


def test_last_physical_token_slot_is_valid():
    cache, pages, mapping = inputs()
    mapping[32] = 11  # Three four-token pages contain slots [0, 12).
    ENSURE(cache, pages, 4, logical_to_device=mapping, physical_num_pages=3)
    assert cache.c4_flat_token_ids[0].item() == 11
