// Diagnostic only. Production dispatch never selects this kernel.
#include "hisparse.cuh"

namespace sglang {
SGL_DEVICE unsigned long long segment_now() {
  unsigned long long value;
  // Compiler memory barrier, NOT a GPU fence. Final SASS must be reviewed.
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

// These are candidate timer boundaries. cuobjdump review is mandatory.
// The XOR sink is externally recorded so result consumption remains live.
SGL_DEVICE unsigned long long segment_consumed(
    int64_t x, int64_t y, int64_t z, unsigned long long& sink) {
  unsigned long long tick, checksum;
  asm volatile(
      "{ xor.b64 %1, %2, %3; xor.b64 %1, %1, %4; "
      "mov.u64 %0, %%globaltimer; }"
      : "=l"(tick), "=&l"(checksum)
      : "l"(x), "l"(y), "l"(z)
      : "memory");
  sink ^= checksum;
  return tick;
}

template <int BLOCK_SIZE, bool Trace>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void copy_cache_segments_kernel(
    const int64_t* __restrict__ miss_src_locs,
    const int32_t* __restrict__ miss_dst_locs,
    const int32_t* __restrict__ miss_counts,
    const int32_t* __restrict__ num_real_reqs,
    const void* __restrict__ host_cache_k,
    const void* __restrict__ host_cache_v,
    void* __restrict__ device_buffer_k,
    void* __restrict__ device_buffer_v,
    int64_t plan_stride,
    int64_t item_size_bytes,
    int64_t* __restrict__ records) {
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  const int warp_global = blockIdx.x * NUM_WARPS + threadIdx.x / WARP_SIZE;
  const int total_warps = gridDim.x * NUM_WARPS;
  const int real = num_real_reqs[0];

  // Warp-sized windows amortize the miss_counts loads; warps then round-robin
  // the flattened (request, miss) space, as in the production kernel.
  unsigned long long begin = 0;
  unsigned long long prep_sum = 0, issue_sum = 0, wait_sum = 0, store_sum = 0;
  unsigned long long sink = 0;
  int bad_order = 0;
  int copied = 0, visits = 0;
  if constexpr (Trace) begin = segment_now();
  int start = 0;  // flat index of the current request's first miss
  for (int base = 0; base < real; base += WARP_SIZE) {
    const int r_lane = base + lane_id;
    const int cnt_lane = (r_lane < real) ? miss_counts[r_lane] : 0;
    const int window = (real - base < WARP_SIZE) ? (real - base) : WARP_SIZE;
    for (int j = 0; j < window; ++j) {
      if constexpr (Trace) ++visits;
      const int cnt = __shfl_sync(FULL_WARP_MASK, cnt_lane, j);
      if (cnt == 0) continue;
      int m0 = (warp_global - start) % total_warps;
      if (m0 < 0) m0 += total_warps;
      const int64_t r = base + j;
      const int64_t* src_row = miss_src_locs + r * plan_stride;
      const int32_t* dst_row = miss_dst_locs + r * plan_stride;
      for (int m = m0; m < cnt; m += total_warps) {
        if constexpr (!Trace) {
          copy_miss_item<true, true, 0>(
              lane_id, host_cache_k, host_cache_v, device_buffer_k, device_buffer_v,
              src_row[m], static_cast<int64_t>(dst_row[m]), item_size_bytes);
        } else {
          const auto t0 = segment_now();
          const int32_t src_index = static_cast<int32_t>(src_row[m]);
          const int32_t dst_index = dst_row[m];
          using namespace device::hisparse;
          const auto src = get_pointer_paged(const_cast<void*>(host_cache_k), src_index);
          const auto dst = get_pointer_paged(device_buffer_k, dst_index);
          const auto tail_src = lane_id < 8 ? src.value_ptr + 64 + lane_id : src.scale_ptr;
          const auto tail_dst = lane_id < 8 ? dst.value_ptr + 64 + lane_id : dst.scale_ptr;
          const auto src0 = src.value_ptr + lane_id;
          const auto src1 = src.value_ptr + lane_id + 32;
          const auto dst0 = dst.value_ptr + lane_id;
          const auto dst1 = dst.value_ptr + lane_id + 32;
          // Address-dependent predicate makes all six address calculations live
          // before T1. Valid tensor addresses have a nonzero bitwise OR.
          const auto addresses = reinterpret_cast<unsigned long long>(src0) |
              reinterpret_cast<unsigned long long>(src1) |
              reinterpret_cast<unsigned long long>(dst0) |
              reinterpret_cast<unsigned long long>(dst1) |
              reinterpret_cast<unsigned long long>(tail_src) |
              reinterpret_cast<unsigned long long>(tail_dst);
          unsigned long long t1;
          asm volatile("{ .reg .pred p; setp.ne.u64 p, %1, 0; "
                       "mov.u64 %0, 0; @p mov.u64 %0, %%globaltimer; }"
                       : "=l"(t1) : "l"(addresses) : "memory");
          const auto a = load_item_word<0>(src0);
          const auto b = load_item_word<0>(src1);
          const auto c = load_item_word<0>(tail_src);
          const auto t2 = segment_now();
          const auto t3 = segment_consumed(a, b, c, sink);
          *dst0 = a;
          *dst1 = b;
          *tail_dst = c;
          const auto t4 = segment_now();
          prep_sum += t1 - t0;
          issue_sum += t2 - t1;
          wait_sum += t3 - t2;
          store_sum += t4 - t3;
          bad_order += !(t0 <= t1 && t1 <= t2 && t2 <= t3 && t3 <= t4);
          ++copied;
        }
      }
      start += cnt;
    }
  }
  if constexpr (Trace) {
    const auto end = segment_now();
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if (lane_id == 0) {
      // One row per warp; guard rows are allocated by the caller.
      auto row = records + warp_global * 15;
      row[0] = blockIdx.x;
      row[1] = threadIdx.x / WARP_SIZE;
      row[2] = warp_global;
      row[3] = sm;
      row[4] = begin;
      row[5] = end;
      row[6] = prep_sum + issue_sum + wait_sum + store_sum;
      row[7] = copied;
      row[8] = visits;
      row[9] = prep_sum;
      row[10] = issue_sum;
      row[11] = wait_sum;
      row[12] = store_sum;
      row[13] = static_cast<int64_t>(sink);
      row[14] = bad_order;
    }
  }
}

template <int BLOCK_SIZE, bool Trace>
void copy_cache_segments(
    tvm::ffi::TensorView src,
    tvm::ffi::TensorView dst,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView real,
    tvm::ffi::TensorView source,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView records,
    int64_t blocks) {
  using namespace host;
  RuntimeCheck(blocks == 4 || blocks == 8, "Latency grid must be 4 or 8");
  RuntimeCheck(src.strides()[0] == dst.strides()[0], "Plan row strides differ");
  const auto device = LaunchKernel::resolve_device(src.device());
  LaunchKernel(blocks, BLOCK_SIZE, device)(
      copy_cache_segments_kernel<BLOCK_SIZE, Trace>,
      static_cast<const int64_t*>(src.data_ptr()),
      static_cast<const int32_t*>(dst.data_ptr()),
      static_cast<const int32_t*>(counts.data_ptr()),
      static_cast<const int32_t*>(real.data_ptr()),
      source.data_ptr(),
      static_cast<const void*>(nullptr),
      output.data_ptr(),
      static_cast<void*>(nullptr),
      src.strides()[0],
      int64_t(584),
      static_cast<int64_t*>(records.data_ptr()));
}
}  // namespace sglang
