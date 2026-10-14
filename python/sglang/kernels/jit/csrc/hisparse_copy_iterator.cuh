#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>

namespace sglang {
namespace hisparse_copy_diagnostic {

// All lanes in a warp share cursor state except lane_count. Each warp owns
// flat indices w, w + total_warps, ... exactly as copy_cache_planned_kernel.
// Keeping the cursor across calls permits staging across request boundaries.
struct MissIterator {
  const int32_t* counts;
  int real;
  int step;
  int lane;
  int flat;
  int row = -1;
  int start = 0;
  int end = 0;
  int lane_count = 0;

  SGL_DEVICE MissIterator(const int32_t* counts_, int real_, int step_, int lane_, int warp_)
      : counts(counts_), real(real_), step(step_), lane(lane_), flat(warp_) {}

  SGL_DEVICE bool next(int& request, int& miss) {
    if (row >= real) return false;
    while (flat >= end) {
      ++row;
      if (row >= real) return false;
      if ((row & 31) == 0) {
        const int r = row + lane;
        lane_count = r < real ? counts[r] : 0;
      }
      const int count = __shfl_sync(0xffffffffu, lane_count, row & 31);
      start = end;
      end += count;
    }
    request = row;
    miss = flat - start;
    flat += step;
    return true;
  }
};

struct KVPlan {
  int32_t source;
  int32_t destination;
};

struct PendingKV {
  int64_t prefix0;
  int64_t prefix1;
  int64_t tail;
  int32_t destination;
};

// Same scalar loads, lane mapping and C4 layout as transfer_item<0>.
// Source code order is an experimental intervention, not a guarantee of SASS
// scheduling: inspect the generated artifact before doing performance trials.
SGL_DEVICE PendingKV read_kv(const void* source, int32_t src, int32_t dst, int lane) {
  using namespace device::hisparse;
  const auto pointers = get_pointer_paged(const_cast<void*>(source), src);
  const auto v0 = load_item_word<0>(pointers.value_ptr + lane);
  const auto v1 = load_item_word<0>(pointers.value_ptr + lane + 32);
  const auto tail = load_item_word<0>(lane < 8 ? pointers.value_ptr + 64 + lane : pointers.scale_ptr);
  return {v0, v1, tail, dst};
}

SGL_DEVICE void write_kv(void* destination, const PendingKV& item, int lane) {
  using namespace device::hisparse;
  const auto pointers = get_pointer_paged(destination, item.destination);
  pointers.value_ptr[lane] = item.prefix0;
  pointers.value_ptr[lane + 32] = item.prefix1;
  *(lane < 8 ? pointers.value_ptr + 64 + lane : pointers.scale_ptr) = item.tail;
}

}  // namespace hisparse_copy_diagnostic

template <int BLOCK_SIZE, int DEPTH>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void copy_cache_planned_iterator_kernel(
    const int64_t* __restrict__ sources,
    const int32_t* __restrict__ destinations,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ num_real_reqs,
    const void* __restrict__ host_cache,
    void* __restrict__ device_buffer,
    int64_t stride) {
  static_assert(DEPTH == 1 || DEPTH == 2);
  static_assert(BLOCK_SIZE % 32 == 0);
  using namespace hisparse_copy_diagnostic;
  const int lane = threadIdx.x & 31;
  const int warp = blockIdx.x * (BLOCK_SIZE / 32) + threadIdx.x / 32;
  const int warps = gridDim.x * (BLOCK_SIZE / 32);
  MissIterator iterator(counts, num_real_reqs[0], warps, lane, warp);
  while (true) {
    // Finish both cursor walks (including SHFL) and GPU plan reads before
    // issuing host loads. Interleaving next(B) with load(A) caused the old D2
    // SHFL to wait on the scoreboard barrier also used by A's source loads.
    KVPlan plans[DEPTH];
    int filled = 0;
#pragma unroll
    for (int slot = 0; slot < DEPTH; ++slot) {
      int request, miss;
      if (iterator.next(request, miss)) {
        const auto offset = static_cast<int64_t>(request) * stride + miss;
        plans[slot] = {static_cast<int32_t>(sources[offset]), destinations[offset]};
        ++filled;
      }
    }
    if (filled == 0) break;
    if constexpr (DEPTH == 2) {
      // Branch before either source read, not between the two reads. There is
      // no iterator/SHFL or per-item validity branch inside the full-pair body.
      // Actual instruction scheduling and barrier reuse still need SASS review.
      if (filled == 2) {
        const auto a = read_kv(host_cache, plans[0].source, plans[0].destination, lane);
        const auto b = read_kv(host_cache, plans[1].source, plans[1].destination, lane);
        write_kv(device_buffer, a, lane);
        write_kv(device_buffer, b, lane);
        continue;
      }
    }
    // D1 and the odd D2 tail share the same single-item path.
    const auto a = read_kv(host_cache, plans[0].source, plans[0].destination, lane);
    write_kv(device_buffer, a, lane);
  }
}

template <int BLOCK_SIZE, int DEPTH>
void copy_cache_planned_iterator(
    tvm::ffi::TensorView sources,
    tvm::ffi::TensorView destinations,
    tvm::ffi::TensorView counts,
    tvm::ffi::TensorView num_real_reqs,
    tvm::ffi::TensorView host_cache,
    tvm::ffi::TensorView device_buffer,
    int64_t num_blocks) {
  using namespace host;
  const auto stride = sources.strides()[0];
  RuntimeCheck(stride == destinations.strides()[0], "Iterator source/destination row strides differ");
  const auto device = LaunchKernel::resolve_device(sources.device());
  LaunchKernel(num_blocks, BLOCK_SIZE, device)(
      copy_cache_planned_iterator_kernel<BLOCK_SIZE, DEPTH>,
      static_cast<const int64_t*>(sources.data_ptr()),
      static_cast<const int32_t*>(destinations.data_ptr()),
      static_cast<const int32_t*>(counts.data_ptr()),
      static_cast<const int32_t*>(num_real_reqs.data_ptr()),
      host_cache.data_ptr(),
      device_buffer.data_ptr(),
      stride);
}

}  // namespace sglang
