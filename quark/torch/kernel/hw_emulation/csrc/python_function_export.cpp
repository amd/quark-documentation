//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ATen/ATen.h>
#include <torch/extension.h>

torch::Tensor fake_quantize_per_tensor_affine(const torch::Tensor& input, const torch::Tensor& scale, const torch::Tensor& zero_point, int64_t quant_min, int64_t quant_max, int64_t round_mode);
at::Tensor quant_fp8_e4m3(at::Tensor inputs);
at::Tensor dequant_fp8_e4m3(at::Tensor inputs);
at::Tensor quant_dequant_fp8_e4m3(at::Tensor inputs);
at::Tensor quant_dequant_fp8_e4m3_only_cuda_runtime(at::Tensor inputs);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fake_quantize_per_tensor_affine", &fake_quantize_per_tensor_affine, "fake_quantize_per_tensor_affine function",
        py::arg("inputs"),
        py::arg("scale"),
        py::arg("zero_point"),
        py::arg("quant_min"),
        py::arg("quant_max"),
        py::arg("round_mode")
  );

  m.def("quant_fp8_e4m3", &quant_fp8_e4m3, "quant float to fp8 e4m3",
        py::arg("inputs"));

  m.def("dequant_fp8_e4m3", &dequant_fp8_e4m3, "dequant uint8 fp8 e4m3 to float",
        py::arg("inputs"));

  m.def("quant_dequant_fp8_e4m3", &quant_dequant_fp8_e4m3, "quant dequant float to uint8 fp8 e4m3 to float",
        py::arg("inputs"));

#if IS_CUDA_RUNTIME
  m.def("quant_dequant_fp8_e4m3_only_cuda_runtime", &quant_dequant_fp8_e4m3_only_cuda_runtime, "quant dequant float to uint8 fp8 e4m3 to float",
        py::arg("inputs"));
#endif
}
