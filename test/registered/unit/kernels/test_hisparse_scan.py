"""Exercise the actual HiSparse scan and its call sites without a GPU.

The C++ harness runs one CPU thread per lane with barrier-backed shuffle
intrinsics. It compiles the scan function and both warp-0 blocks extracted from
hisparse.cuh, so the CUDA/ROCm bounds used by the kernel are tested directly.
This checks scan arithmetic/footprint, not GPU scheduling or memory transfers.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SOURCE = (
    Path(__file__).resolve().parents[4] / "python/sglang/kernels/jit/csrc/hisparse.cuh"
)


def block_at(source, start):
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


HARNESS = r"""
#include <algorithm>
#include <barrier>
#include <iostream>
#include <thread>
#include <vector>
#define __device__
#define __forceinline__ inline
using std::min;
constexpr unsigned long long FULL_WARP_MASK = ~0ull;
thread_local int test_lane;
std::barrier sync_lanes(WARP_SIZE);
int lane_values[WARP_SIZE];
int shuffle(int value, int source_lane) {
  lane_values[test_lane] = value;
  sync_lanes.arrive_and_wait();
  int result = lane_values[source_lane];
  sync_lanes.arrive_and_wait();
  return result;
}
int __shfl_up_sync(unsigned long long, int value, int delta) {
  return shuffle(value, test_lane >= delta ? test_lane - delta : test_lane);
}
int __shfl_sync(unsigned long long, int value, int source_lane) {
  return shuffle(value, source_lane);
}
// SCAN_FUNCTION
int main(int argc, char** argv) {
  const int NUM_WARPS = std::stoi(argv[1]) / WARP_SIZE;
  const int NUM_BUFFER_CHUNKS = 3072 / WARP_SIZE;
  const int NUM_TOKEN_CHUNKS = 512 / WARP_SIZE;
  bool miss_phase = std::stoi(argv[2]);
  const int chunks = miss_phase ? NUM_TOKEN_CHUNKS : NUM_BUFFER_CHUNKS;
  std::vector<int> s_chunk_offset(NUM_BUFFER_CHUNKS + 1, 0);
  std::vector<int> s_evict_chunk_offset(NUM_BUFFER_CHUNKS + 1, 0);
  int s_total_hits = 0;
  bool failed = false;
  std::vector<std::thread> lanes;
  for (int lane_id = 0; lane_id < WARP_SIZE; ++lane_id) {
    lanes.emplace_back([&, lane_id]() {
      test_lane = lane_id;
      const int tid = lane_id, warp_id = 0;
      int total_hit_count = 0, total_evict_count = 0, total_misses = 0;
      for (int iter = 0; iter * NUM_WARPS < chunks; ++iter) {
        const int chunk_idx = iter * NUM_WARPS;
        const int end = min(chunk_idx + NUM_WARPS, chunks);
        if (lane_id == 0) {
          // Other warps produced one count each in the current tile only.
          for (int j = chunk_idx; j < end; ++j) {
            s_chunk_offset[j + 1] = 1;
            s_evict_chunk_offset[j + 1] = WARP_SIZE - 1;
          }
        }
        sync_lanes.arrive_and_wait();
        // Kernel code uses raw shared-memory pointers.
        int* hits = s_chunk_offset.data();
        int* evicts = s_evict_chunk_offset.data();
        {
          int* s_chunk_offset = hits;
          int* s_evict_chunk_offset = evicts;
          if (miss_phase) {
            // MISS_CALL_SITE
          } else {
            // HIT_CALL_SITE
          }
        }
        sync_lanes.arrive_and_wait();
        if (lane_id == 0) {
          const int total = miss_phase ? total_misses : total_hit_count;
          if (total != end || (!miss_phase && total_evict_count != end * (WARP_SIZE - 1))) {
            std::cerr << "wrong accumulator at iter=" << iter << " actual=" << total << " expected=" << end << '\n';
            failed = true;
          }
          for (int j = 0; j <= chunks; ++j) {
            const int expected = j <= end ? j : 0;
            if (s_chunk_offset[j] != expected || (!miss_phase && s_evict_chunk_offset[j] != expected * (WARP_SIZE - 1))) {
              std::cerr << "scan overwrote offset " << j << " at iter=" << iter << " past current tile end=" << end << '\n';
              failed = true;
              break;
            }
          }
        }
        sync_lanes.arrive_and_wait();
        if (failed) break;
      }
    });
  }
  for (auto& lane : lanes) lane.join();
  return failed ? 1 : 0;
}
"""


@pytest.fixture(scope="module", params=[32, 64], ids=["cuda-warp32", "rocm-wave64"])
def scan_binary(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++20 compiler required for host shuffle emulator")
    source = SOURCE.read_text()
    function = block_at(
        source, source.index("__device__ __forceinline__ int warp_inclusive_scan")
    )
    first = source.index("if (warp_id == 0)", source.index("int total_hit_count"))
    second = source.index("if (warp_id == 0)", source.index("int total_misses"))
    code = HARNESS.replace("// SCAN_FUNCTION", function)
    code = code.replace("// HIT_CALL_SITE", block_at(source, first))
    code = code.replace("// MISS_CALL_SITE", block_at(source, second))
    root = tmp_path_factory.mktemp("hisparse-scan")
    cpp, binary = root / "scan.cpp", root / "scan"
    cpp.write_text(code)
    flags = [f"-DWARP_SIZE={request.param}"]
    if request.param == 64:
        flags.append("-DUSE_ROCM")
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-pthread",
            "-O2",
            *flags,
            str(cpp),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("block_size", [256, 512, 960, 1024])
@pytest.mark.parametrize("phase", [0, 1], ids=["hits-evictions", "misses"])
def test_scan_does_not_touch_future_tiles(scan_binary, block_size, phase):
    result = subprocess.run(
        [str(scan_binary), str(block_size), str(phase)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
