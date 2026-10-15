// Diagnostic only. Production dispatch never selects this kernel.
#include "hisparse.cuh"

namespace sglang {
SGL_DEVICE unsigned long long latency_now() {
  unsigned long long value;
  // Compiler memory barrier, NOT a GPU fence. Final SASS must be reviewed.
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
  return value;
}

template <int BLOCK_SIZE, bool Trace>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void copy_cache_latency_kernel(
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
  unsigned long long begin = 0, copy_sum = 0;
  int copied = 0, visits = 0;
  if constexpr (Trace) begin = latency_now();
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
        unsigned long long copy_begin = 0;
        if constexpr (Trace) copy_begin = latency_now();
        copy_miss_item<true, true, 0>(
            lane_id,
            host_cache_k,
            host_cache_v,
            device_buffer_k,
            device_buffer_v,
            src_row[m],
            static_cast<int64_t>(dst_row[m]),
            item_size_bytes);
        if constexpr (Trace) {
          copy_sum += latency_now() - copy_begin;
          ++copied;
        }
      }
      start += cnt;
    }
  }
  if constexpr (Trace) {
    const auto end = latency_now();
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if (lane_id == 0) {
      // One row per warp; guard rows are allocated by the caller.
      auto row = records + warp_global * 9;
      row[0] = blockIdx.x;
      row[1] = threadIdx.x / WARP_SIZE;
      row[2] = warp_global;
      row[3] = sm;
      row[4] = begin;
      row[5] = end;
      row[6] = copy_sum;
      row[7] = copied;
      row[8] = visits;
    }
  }
}

template <int BLOCK_SIZE, bool Trace>
void copy_cache_latency(
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
      copy_cache_latency_kernel<BLOCK_SIZE, Trace>,
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
