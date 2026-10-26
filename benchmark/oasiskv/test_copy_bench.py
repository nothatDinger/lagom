"""CPU workload accounting; these tests do not measure GPU performance."""

import unittest

from bench_copy import make_counts


class CopyWorkloadTests(unittest.TestCase):
    def test_exact_total_is_not_the_number_of_grid_groups(self):
        counts = make_counts(8, 7, 8, 128, total=64)
        self.assertEqual(sum(counts), 64)
        self.assertEqual(counts[-8:], [0] * 8)
        self.assertEqual(counts.count(2), 8)
        self.assertEqual(counts.count(1), 48)

    def test_per_head_misses_exclude_graph_padding(self):
        counts = make_counts(8, 7, 8, 128, per_head=64)
        self.assertEqual(sum(counts), 7 * 8 * 64)
        self.assertEqual(counts[-8:], [0] * 8)
        self.assertEqual(make_counts(8, 7, 8, 128, total=0), [0] * 64)

    def test_reject_overflow_and_invalid_live_batch(self):
        for live, kwargs in (
            (9, {"total": 1}),
            (7, {"total": 7169}),
            (7, {"per_head": 129}),
            (7, {"total": -1}),
            (7, {}),
            (7, {"per_head": 1, "total": 64}),
        ):
            with self.subTest(live=live, kwargs=kwargs), self.assertRaises(ValueError):
                make_counts(8, live, 8, 128, **kwargs)


if __name__ == "__main__":
    unittest.main()
