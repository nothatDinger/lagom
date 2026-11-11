"""CPU-only regressions for Lagom's PD state layout and actual copy addressing.

Run with plain Python:
  python -m unittest discover -s test/registered/unit/disaggregation \
      -p test_lagom_pd_state_layout.py

Load production host methods via AST to avoid importing CUDA/Mooncake or
constructing a GPU pool. No transfer/address arithmetic is reimplemented here.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
POOL = "python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py"
CONFIGURATOR = "python/sglang/srt/model_executor/pool_configurator.py"
MOONCAKE = "python/sglang/srt/disaggregation/mooncake/conn.py"
LAYOUT = "python/sglang/srt/disaggregation/state_layout.py"

spec = importlib.util.spec_from_file_location("pd_state_layout", ROOT / LAYOUT)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def function_node(path, name, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if cls is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return copy.deepcopy(next(
        n for n in body if isinstance(n, ast.FunctionDef) and n.name == name
    ))


def load_function(path, name, namespace, cls=None):
    node = function_node(path, name, cls)
    node.decorator_list = []
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / path), "exec"), namespace)
    return namespace[name]


RING_NAMESPACE = {"ONLINE_C128": False}
load_function(POOL, "get_compress_state_write_pad", RING_NAMESPACE)
ring_size = load_function(POOL, "get_compress_state_ring_size", RING_NAMESPACE)
buf_infos = load_function(POOL, "get_state_buf_infos", {}, "DeepSeekV4TokenToKVPool")


def budget_rings(args):
    # Execute the production allocation/budget decision, not a test copy.
    node = function_node(CONFIGURATOR, "__init__", "DSV4PoolConfigurator")
    start = next(i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "execution_width" for t in n.targets))
    end = next(i for i, n in enumerate(node.body[start:], start)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "c128_ring_size" for t in n.targets))
    obj = SimpleNamespace(is_speculative=True)
    namespace = {
        "self": obj, "kvc": SimpleNamespace(server_args=args),
        "dsv4_state_execution_width": layout.dsv4_state_execution_width,
        "get_compress_state_ring_size": ring_size,
    }
    tree = ast.Module(body=node.body[start:end + 1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(ROOT / CONFIGURATOR), "exec"), namespace)
    return obj.c4_ring_size, obj.c128_ring_size


class Buffer:
    ndim = 2

    def __init__(self, ptr, row_bytes, rows):
        self.ptr = ptr
        self.row_bytes = row_bytes
        self.nbytes = row_bytes * rows

    def data_ptr(self):
        return self.ptr

    def __getitem__(self, index):
        assert index == 0
        return SimpleNamespace(nbytes=self.row_bytes)


def pool_metadata(ring, base, pages):
    def state(ptr):
        return SimpleNamespace(
            ratio=4, ring_size=ring,
            kv_score_buffer=SimpleNamespace(kv_score=Buffer(ptr, 4, pages * ring)),
        )

    # Include both the attention and indexer compressor, plus an SWA entry.
    return buf_infos(SimpleNamespace(
        _unified_kv=False,
        swa_kv_pool=SimpleNamespace(kv_buffer=[Buffer(base, 16, pages)]),
        compress_state_pools=[state(base + 10000), None],
        indexer_compress_state_pools=[state(base + 20000), None],
    ))


def sender():
    namespace = {
        # Each test supplies a single contiguous page block. Grouping is
        # unrelated to the byte-stride regression under test.
        "group_concurrent_contiguous": lambda src, dst: ([src], [dst]),
        "validate_state_transfer_layout": layout.validate_state_transfer_layout,
    }
    send = load_function(MOONCAKE, "_send_kvcache_generic", namespace, "MooncakeKVManager")
    calls = []
    obj = SimpleNamespace(
        pp_size=1, is_mla_backend=True, is_hybrid_mla_backend=False,
        enable_custom_mem_pool=False,
        get_mla_kv_ptrs_with_pp=lambda src, dst, state_type: (src, dst, len(src)),
        _transfer_data=lambda session, blocks: calls.extend(blocks) or 0,
    )
    return obj, send, calls


class TestLagomPDStateLayout(unittest.TestCase):
    def test_prefill_budget_matches_decoder_for_configurable_windows(self):
        for w, p, expected_ring in ((7, 1, 16), (7, 2, 16), (7, 4, 32), (7, 7, 32), (3, 4, 16)):
            with self.subTest(w=w, p=p):
                width = w + p + 1
                prefill = SimpleNamespace(
                    is_lagom=False, max_speculative_num_draft_tokens=w + 1,
                    disaggregation_dsv4_state_execution_width=width,
                )
                decode = SimpleNamespace(
                    is_lagom=True, max_speculative_num_draft_tokens=width,
                    disaggregation_dsv4_state_execution_width=None,
                )
                self.assertEqual(budget_rings(prefill), (expected_ring, 256))
                self.assertEqual(budget_rings(prefill), budget_rings(decode))
                self.assertEqual(prefill.max_speculative_num_draft_tokens, w + 1)
                self.assertFalse(prefill.is_lagom)

    def test_ordinary_and_lagom_layouts_keep_previous_defaults(self):
        self.assertEqual(layout.dsv4_state_execution_width(
            is_lagom=False, max_speculative_num_draft_tokens=8), 0)
        for width in (9, 12, 15, 16):
            self.assertEqual(layout.dsv4_state_execution_width(
                is_lagom=True, max_speculative_num_draft_tokens=width), width)

    def test_buffer_metadata_matches_with_different_pool_capacities(self):
        src = pool_metadata(32, 100000, 5)
        dst = pool_metadata(32, 200000, 9)
        self.assertEqual(src[2], [16, 128, 128])
        self.assertEqual(src[2], dst[2])
        self.assertNotEqual(src[1], dst[1])  # Total bytes need not match.
        layout.validate_state_transfer_layout(
            list(zip(src[0], dst[0], src[2])), dst[0], dst[2], state_type="swa"
        )

    def test_old_16_to_32_layout_is_rejected_before_any_dma(self):
        obj, send, calls = sender()
        src, dst = pool_metadata(16, 100000, 5), pool_metadata(32, 200000, 5)
        with self.assertRaisesRegex(RuntimeError, "state layout mismatch"):
            send(obj, "test", src[0], dst[0], src[2], [1], [3], None,
                 state_type="swa", dst_state_item_lens=dst[2])
        self.assertEqual(calls, [])

    def test_nonzero_destination_page_and_ring_wrap_use_matching_strides(self):
        for p in (2, 4, 7):
            ring = ring_size(4, True, 7 + p + 1)
            for prefix in (33, 47, 63, 64, 65):
                with self.subTest(p=p, prefix=prefix):
                    obj, send, calls = sender()
                    src = pool_metadata(ring, 100000, 5)
                    dst = pool_metadata(ring, 200000, 7)
                    send(obj, "test", src[0], dst[0], src[2], [1], [3], None,
                         state_type="swa", dst_state_item_lens=dst[2])
                    self.assertEqual(len(calls), 3)
                    for index in (1, 2):  # Both compressor state buffers.
                        item_bytes = 4 * ring
                        source = bytearray(src[1][index])
                        target = bytearray(dst[1][index])
                        for position in range(prefix - 8, prefix):
                            struct.pack_into("<I", source,
                                             item_bytes + (position % ring) * 4,
                                             position + 1)
                        src_addr, dst_addr, count = calls[index]
                        src_offset = src_addr - src[0][index]
                        dst_offset = dst_addr - dst[0][index]
                        self.assertEqual(dst_offset, 3 * item_bytes)
                        target[dst_offset:dst_offset + count] = source[src_offset:src_offset + count]
                        for position in range(prefix - 8, prefix):
                            self.assertEqual(
                                struct.unpack_from("<I", target,
                                                   3 * item_bytes + (position % ring) * 4)[0],
                                position + 1,
                            )

    def test_metadata_and_buffer_counts_fail_before_dma(self):
        for dst_ptrs, dst_sizes in (([2000], []), ([2000, 3000], [128, 128])):
            obj, send, calls = sender()
            with self.assertRaises(RuntimeError):
                send(obj, "test", [1000], dst_ptrs, [128], [1], [3], None,
                     state_type="swa", dst_state_item_lens=dst_sizes)
            self.assertEqual(calls, [])

    def test_pp_pairing_validates_only_the_selected_destination_stride(self):
        obj, send, calls = sender()
        obj.pp_size = 2
        obj.get_mla_kv_ptrs_with_pp = lambda src, dst, st: (src, dst[1:], len(src))
        send(obj, "test", [1000], [2000, 3000], [128], [1], [3], None,
             state_type="swa", dst_state_item_lens=[64, 128])
        self.assertEqual(calls, [(1128, 3384, 128)])

    def test_state_sender_passes_receiver_metadata_into_validated_copy(self):
        node = function_node(MOONCAKE, "maybe_send_extra", "MooncakeKVManager")
        generic = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr == "_send_kvcache_generic"]
        self.assertTrue(any(
            any(kw.arg == "state_type" for kw in call.keywords)
            and any(kw.arg == "dst_state_item_lens" for kw in call.keywords)
            for call in generic
        ))


if __name__ == "__main__":
    unittest.main()
