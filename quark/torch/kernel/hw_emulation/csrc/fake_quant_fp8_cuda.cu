//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if IS_CUDA_RUNTIME

#include <ATen/ATen.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include "custom_type.h"


template <typename T>
__global__ void quant_dequant_e4m3_kernel(const T *inputs, size_t n, T *outputs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int idx = 4 * tid; idx < 4 * (tid + 1) && idx < n; ++idx) {
    outputs[idx] = static_cast<T>(static_cast<float>(
        static_cast<__nv_fp8_e4m3>(static_cast<float>(inputs[idx]))));
  }
}

at::Tensor quant_dequant_e4m3_cuda(at::Tensor inputs) {
  size_t numel = inputs.numel();
  auto outputs = torch::empty_like(inputs);
  AT_DISPATCH_FLOATING_TYPES(
      inputs.type().scalarType(), "quant_dequant_e4m3_cuda", [&] {
        quant_dequant_e4m3_kernel<<<numel / (BLOCK_SIZE * 4) + 1, BLOCK_SIZE>>>(
            inputs.data_ptr<scalar_t>(), numel, outputs.data_ptr<scalar_t>());
      });
  return outputs;
}

#endif
