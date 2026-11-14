"""Plain-Python layout and allocation-accounting regressions."""
import ast
import collections
import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
source = ROOT / "python/sglang/srt/mem_cache/hisparse_staging.py"
spec = importlib.util.spec_from_file_location("layout_under_test", source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
Layout = module.HiSparseStagingLayout


class TestStagingLayout(unittest.TestCase):
    def test_slots_follow_sparse_ordinals_without_hardcoding_21(self):
        for layers in (1, 3, 21):
            layout = Layout(9, 12, 512, layers, 4)
            self.assertEqual(layout.slot_count, layers)
            self.assertEqual([layout.slot_for_layer(i) for i in range(layers)], list(range(layers)))
        self.assertEqual(Layout(9, 12, 512, 21, 4, 24).slot_count, 24)

    def test_configurable_windows_do_not_require_a_larger_hot_cache(self):
        for p in (1, 2, 4, 7):
            layout = Layout(9, 8 + p, 512, 21, p, 21)
            self.assertEqual(layout.verify_capacity, (8 + p) * 512)
            self.assertEqual(layout.forecast_capacity, p * 512)

    def test_full_step_rejects_overwriting_unconsumed_layer_forecasts(self):
        for slots in (1, 2, 20):
            with self.assertRaisesRegex(ValueError, "rolling"):
                Layout(9, 12, 512, 21, 4, slots)

    def test_invalid_geometry(self):
        for change in (
            {"prefetch_slots": 0}, {"prefetch_slots": True},
            {"prediction_steps": -1}, {"prediction_steps": 12},
            {"verify_width": 17}, {"page_size": 32},
        ):
            args = dict(request_slots=9, verify_width=12, top_k=512,
                        sparse_layers=21, prediction_steps=4)
            args.update(change)
            with self.assertRaises(ValueError):
                Layout(**args)

    def test_verification_arena_is_not_replicated_for_every_layer(self):
        one = Layout(9, 12, 512, 1, 4)
        full = Layout(9, 12, 512, 21, 4)
        expected = 20 * (math.ceil(9 * 4 * 512 / 64) + 1) * 37440
        self.assertEqual(full.payload_bytes(37440) - one.payload_bytes(37440), expected)
        self.assertEqual(Layout(9, 8, 512, 21).slot_count, 0)

    def test_budget_matches_actual_constructor_allocations(self):
        # Execute the production constructor with recording allocations.
        path = ROOT / "python/sglang/kernels/ops/kvcache/hisparse_staging.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "HiSparseVerificationStaging")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        tree = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            init,
        ], type_ignores=[])
        for layers, p in ((1, 0), (21, 0), (21, 4), (21, 7)):
            allocated = []
            def allocate(shape, fill, *, dtype, device):
                result = SimpleNamespace(nbytes=math.prod(shape) * {"i32": 4, "i64": 8, "byte": 1}[dtype])
                allocated.append(result)
                return result
            pool = SimpleNamespace(
                device="record-only",
                create_buffer=lambda *, num_pages: allocate(
                    (num_pages, 37440), 0, dtype="byte", device="record-only"),
            )
            namespace = {
                "torch": SimpleNamespace(full=allocate, int32="i32", int64="i64"),
                "VerificationPlan": collections.namedtuple(
                    "VerificationPlan", "table sources kinds demand tokens counts"),
                "_module": lambda *args: None,
            }
            exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
            layout = Layout(9, 8 + p, 512, layers, p)
            namespace["__init__"](SimpleNamespace(), layout=layout, pool=pool, hot_slots=512)
            self.assertEqual(sum(x.nbytes for x in allocated),
                             layout.payload_bytes(37440) + layout.workspace_bytes())


if __name__ == "__main__":
    unittest.main()
