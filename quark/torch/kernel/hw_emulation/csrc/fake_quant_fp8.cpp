//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if IS_CUDA_RUNTIME

#include <ATen/ATen.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

at::Tensor quant_dequant_e4m3_cuda(at::Tensor inputs);
at::Tensor quant_dequant_fp8_e4m3_only_cuda_runtime(at::Tensor inputs) {
  if (inputs.is_cuda()) {
    return quant_dequant_e4m3_cuda(inputs.contiguous());
  } else {
    TORCH_CHECK(inputs.dtype() == at::ScalarType::Float);
    TORCH_CHECK(inputs.is_contiguous());
    auto out = at::zeros_like(inputs);
    for (int i = 0; i < inputs.numel(); ++i) {
      out.data_ptr<float>()[i] = static_cast<float>(
          static_cast<__nv_fp8_e4m3>(inputs.data_ptr<float>()[i]));
    }
    return out;
  }
}

#endif
