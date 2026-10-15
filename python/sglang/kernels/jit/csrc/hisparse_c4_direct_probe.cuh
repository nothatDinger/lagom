#pragma once

// Diagnostic only: adjacent C4 indices, no plans/counts and no serving changes.
#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>
#include <stdint.h>

namespace sglang {

__global__ __launch_bounds__(1024, 1) void hisparse_c4_direct_kernel(
    const void* __restrict__ source,
    void* __restrict__ output,
    int32_t source_start,
    int32_t destination_start,
    int32_t count) {
  const int warp = blockIdx.x * 32 + threadIdx.x / 32;
  // Exactly the adjacent planned kernel's ordered flat items for each warp.
  // Keep a single KV per iteration; do not introduce a new prefetch stage.
#pragma unroll 1
  for (int flat = warp; flat < count; flat += 128) {
    device::hisparse::transfer_item<0>(
        output, const_cast<void*>(source), destination_start + flat, source_start + flat);
  }
}

void hisparse_c4_direct(
    tvm::ffi::TensorView source,
    tvm::ffi::TensorView output,
    int64_t source_start,
    int64_t destination_start,
    int64_t count) {
  using namespace host;
  constexpr int64_t page_bytes = ::sglang::device::hisparse::kPageBytes;
  auto source_pages = SymbolicSize{"source_pages"};
  auto output_pages = SymbolicSize{"output_pages"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({source_pages, page_bytes}).with_dtype<uint8_t>().verify(source);
  TensorMatcher({output_pages, page_bytes}).with_dtype<uint8_t>().with_device(device).verify(output);
  RuntimeCheck(
      source_start >= 0 && destination_start >= 0 && count >= 0 && count <= INT32_MAX - 128 &&
          source_start <= INT32_MAX && destination_start <= INT32_MAX &&
          source_start + count <= INT32_MAX && destination_start + count <= INT32_MAX,
      "Direct C4 index range must fit int32 and the warp step");
  RuntimeCheck(
      source_start + count <= source_pages.unwrap() * 64 &&
          destination_start + count <= output_pages.unwrap() * 64,
      "Direct C4 indices leave the source/output allocation");
  RuntimeCheck(
      ((reinterpret_cast<uintptr_t>(source.data_ptr()) | reinterpret_cast<uintptr_t>(output.data_ptr())) & 7) == 0,
      "Direct C4 requires 8-byte aligned allocations");
  LaunchKernel(4, 1024, device.unwrap())(
      hisparse_c4_direct_kernel, static_cast<const void*>(source.data_ptr()), output.data_ptr(),
      static_cast<int32_t>(source_start), static_cast<int32_t>(destination_start), static_cast<int32_t>(count));
}

}  // namespace sglang
