"""Capacity checks must run before paged allocation kernels."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _allocator_with_pages(num_pages: int, page_size: int = 4):
    return SimpleNamespace(
        page_size=page_size,
        device="cpu",
        need_sort=False,
        debug_mode=False,
        free_pages=torch.arange(1, num_pages + 1, dtype=torch.int64),
    )


class TestPagedAllocatorCapacity(unittest.TestCase):
    def test_extend_exhaustion_does_not_launch_kernel(self):
        allocator = _allocator_with_pages(2)
        prefix_lens = torch.tensor([4, 4, 4], dtype=torch.int64)
        seq_lens = torch.tensor([5, 5, 5], dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_extend_kernel"
        ) as kernel:
            result = PagedTokenToKVPoolAllocator.alloc_extend(
                allocator,
                prefix_lens=prefix_lens,
                prefix_lens_cpu=prefix_lens,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens,
                last_loc=torch.tensor([4, 8, 12], dtype=torch.int64),
                extend_num_tokens=3,
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertEqual(len(allocator.free_pages), 2)

    def test_decode_exhaustion_does_not_launch_kernel(self):
        allocator = _allocator_with_pages(1)
        seq_lens = torch.tensor([5, 5], dtype=torch.int64)

        with patch(
            "sglang.srt.mem_cache.allocator.paged.alloc_decode_kernel"
        ) as kernel:
            result = PagedTokenToKVPoolAllocator.alloc_decode(
                allocator,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens,
                last_loc=torch.tensor([3, 7], dtype=torch.int64),
            )

        self.assertIsNone(result)
        kernel.__getitem__.assert_not_called()
        self.assertEqual(len(allocator.free_pages), 1)


if __name__ == "__main__":
    unittest.main()
