#pragma once

// Opt-in diagnostic only. No serving code imports or calls this kernel.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <stdint.h>

namespace sglang {

__global__ __launch_bounds__(1024, 1) void hisparse_dependent_read_kernel(
    const int64_t* source,
    const int64_t* starts,
    int64_t* output,
    int64_t nodes,
    int64_t stride_words,
    int64_t steps,
    int warps) {
  const int warp = threadIdx.x / 32;
  if ((threadIdx.x % 32) != 0 || warp >= warps) return;
  const int chain = blockIdx.x * warps + warp;
  uint64_t index = static_cast<uint64_t>(starts[chain]);
  uint64_t checksum = 0;
  uint32_t smid, nsmid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
  // One dependent 8-byte load per active warp. The next address is unknown
  // until this load returns; extra independent chains are in other warps.
  // .cv re-fetches system-memory lines on repeat traversals (PTX semantics).
#pragma unroll 1
  for (int64_t step = 0; step < steps; ++step) {
    if (index >= static_cast<uint64_t>(nodes)) {
      output[chain * 4] = -2;
      return;
    }
    const int64_t* address = source + index * stride_words;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(index) : "l"(address) : "memory");
    checksum += index;
  }
  output[chain * 4] = static_cast<int64_t>(index);
  output[chain * 4 + 1] = static_cast<int64_t>(checksum);
  output[chain * 4 + 2] = smid;
  output[chain * 4 + 3] = nsmid;
}

void hisparse_dependent_read(
    tvm::ffi::TensorView source,
    tvm::ffi::TensorView starts,
    tvm::ffi::TensorView output,
    int64_t nodes,
    int64_t stride_words,
    int64_t steps,
    int64_t blocks,
    int64_t warps) {
  using namespace host;
  auto words = SymbolicSize{"words"};
  auto chains = SymbolicSize{"chains"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({words}).with_dtype<int64_t>().verify(source);
  TensorMatcher({chains}).with_dtype<int64_t>().with_device(device).verify(starts);
  TensorMatcher({chains, 4}).with_dtype<int64_t>().with_device(device).verify(output);
  RuntimeCheck(blocks >= 1 && blocks <= 4 && warps >= 1 && warps <= 32, "Invalid read-probe launch geometry");
  RuntimeCheck(nodes > 0 && nodes <= (int64_t{1} << 30), "Invalid node count");
  RuntimeCheck(stride_words >= 1 && stride_words <= 512, "Invalid node stride");
  RuntimeCheck(words.unwrap() >= nodes * stride_words, "Source too small");
  RuntimeCheck(chains.unwrap() == blocks * warps, "Wrong start/output chain count");
  RuntimeCheck(steps > 0 && steps <= nodes, "Invalid step count");
  LaunchKernel(blocks, 1024, device.unwrap())(
      hisparse_dependent_read_kernel,
      static_cast<const int64_t*>(source.data_ptr()),
      static_cast<const int64_t*>(starts.data_ptr()),
      static_cast<int64_t*>(output.data_ptr()),
      nodes,
      stride_words,
      steps,
      static_cast<int>(warps));
}

}  // namespace sglang
