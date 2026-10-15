#pragma once

// Diagnostic baseline only. Serving kernels and their JIT inputs are unchanged.
#include <sgl_kernel/deepseek_v4/kvcacheio.cuh>
#include <stdint.h>

namespace sglang {

__global__ __launch_bounds__(1024, 1) void hisparse_linear_copy_kernel(
    const int64_t* __restrict__ source, int64_t* __restrict__ output, int64_t words) {
  const int64_t warp = blockIdx.x * 32 + threadIdx.x / 32;
  const int64_t lane = threadIdx.x % 32;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * 32 * 96;
  // Three scalar 64-bit loads per lane, staged before their stores. The
  // addresses are linear, without KV plans, C4 splitting or a sparse target.
  // Reuse the production legacy C++ load helper, not a new vector/cache policy.
  for (int64_t base = warp * 96; base < words; base += stride) {
    int64_t values[3] = {};
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      const int64_t i = base + lane + p * 32;
      if (i < words) values[p] = device::hisparse::load_item_word<0>(source + i);
    }
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      const int64_t i = base + lane + p * 32;
      if (i < words) output[i] = values[p];
    }
  }
}

void hisparse_linear_copy(
    tvm::ffi::TensorView source, tvm::ffi::TensorView output, int64_t blocks) {
  using namespace host;
  auto bytes = SymbolicSize{"bytes"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({bytes}).with_dtype<uint8_t>().verify(source);
  TensorMatcher({bytes}).with_dtype<uint8_t>().with_device(device).verify(output);
  RuntimeCheck(bytes.unwrap() > 0 && bytes.unwrap() % 8 == 0, "Require positive whole 64-bit words");
  RuntimeCheck(blocks == 4 || blocks == 8, "Linear diagnostic fixes CTA4/CTA8");
  RuntimeCheck(
      ((reinterpret_cast<uintptr_t>(source.data_ptr()) | reinterpret_cast<uintptr_t>(output.data_ptr())) & 7) == 0,
      "Linear diagnostic requires 8-byte alignment");
  LaunchKernel(blocks, 1024, device.unwrap())(
      hisparse_linear_copy_kernel,
      static_cast<const int64_t*>(source.data_ptr()),
      static_cast<int64_t*>(output.data_ptr()),
      bytes.unwrap() / 8);
}

}  // namespace sglang
