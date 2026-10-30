import unittest

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    get_compress_state_write_pad,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCompressStateWritePad(CustomTestCase):
    """The pad bounds how many speculative draft tokens a compress-state ring can serve.

    Mirrors `mtp_pad` in `c_plan.cuh`; `DSV4PoolConfigurator` rejects a larger draft
    count at startup.
    """

    def test_pad_is_zero_without_speculation(self):
        """A non-speculative ring is exactly one window wide: nothing rolls back."""
        for compress_ratio in (4, 128):
            ring_size = get_compress_state_ring_size(compress_ratio, False)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), 0
                )

    def test_pad_matches_speculative_ring_capacity(self):
        """`ring_size - window_size + 2`, with window = 2*cr for the overlapping c4."""
        for compress_ratio, expected in ((4, 10), (128, 130)):
            ring_size = get_compress_state_ring_size(compress_ratio, True)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), expected
                )

    def test_pad_is_zero_for_rings_below_one_window(self):
        """Online c128 collapses the ring to 1; the pad must clamp instead of going
        negative."""
        self.assertEqual(get_compress_state_write_pad(128, 1), 0)

    def test_ring_grows_for_wide_verify_and_preserves_existing_widths(self):
        for width, expected in (
            (6, 16),
            (10, 16),
            (11, 32),
            (12, 32),
            (26, 32),
            (27, 64),
        ):
            with self.subTest(width=width):
                ring = get_compress_state_ring_size(4, True, width)
                self.assertEqual(ring, expected)
                self.assertEqual(256 % ring, 0)
                self.assertGreaterEqual(get_compress_state_write_pad(4, ring), width)
                self.assertEqual(get_compress_state_ring_size(128, True, width), 256)

    def test_ring_rejects_a_width_that_cannot_fit_the_swa_page(self):
        for page_size, width in ((256, 251), (128, 123)):
            with self.subTest(page_size=page_size):
                with self.assertRaisesRegex(ValueError, "must divide swa_page_size"):
                    get_compress_state_ring_size(
                        4, True, width, swa_page_size=page_size
                    )

    def test_request_scoped_c128_does_not_shrink_to_npu_swa_page(self):
        self.assertEqual(
            get_compress_state_ring_size(4, True, 12, swa_page_size=128), 32
        )
        self.assertEqual(
            get_compress_state_ring_size(128, True, 12, swa_page_size=128), 256
        )


if __name__ == "__main__":
    unittest.main()
