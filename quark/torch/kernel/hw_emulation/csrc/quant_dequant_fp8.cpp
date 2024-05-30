//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ATen/ATen.h>
#include <torch/extension.h>
#include "custom_type.h"

at::Tensor quant_fp8_e4m3_cuda(at::Tensor inputs);
at::Tensor dequant_fp8_e4m3_cuda(at::Tensor inputs);
at::Tensor quant_dequant_fp8_e4m3_cuda(at::Tensor inputs);


uint8_t double_to_fp8(const double x, const saturation_t saturate, const fp8_interpretation_t fp8_interpretation);
uint16_t fp8_to_halfraw(const uint8_t x, const fp8_interpretation_t fp8_interpretation);
float halfraw_to_float(const uint16_t x);

at::Tensor quant_fp8_e4m3(at::Tensor inputs) {

    if (inputs.is_cuda()) {
        return quant_fp8_e4m3_cuda(inputs.contiguous());
    } else {
        TORCH_CHECK(inputs.dtype() == at::ScalarType::Float);
        TORCH_CHECK(inputs.is_contiguous());
        auto out = at::zeros_like(inputs, torch::kUInt8);
        for (int i = 0; i < inputs.numel(); ++i) {
            double tmp = static_cast<double>(inputs.data_ptr<float>()[i]);
            uint8_t tmp_u8 = double_to_fp8(
                tmp,
                saturation_t::NOSAT,
                fp8_interpretation_t::E4M3
            );
            out.data_ptr<uint8_t>()[i] = static_cast<uint8_t>(tmp_u8);
        }
        return out;
    }
}

at::Tensor dequant_fp8_e4m3(at::Tensor inputs) {

    if (inputs.is_cuda()) {
        return dequant_fp8_e4m3_cuda(inputs.contiguous());
    } else {
        TORCH_CHECK(inputs.dtype() == torch::kUInt8);
        TORCH_CHECK(inputs.is_contiguous());
        auto out = at::zeros_like(inputs, torch::kFloat);
        for (int i = 0; i < inputs.numel(); ++i) {
            auto halfraw = fp8_to_halfraw(inputs.data_ptr<uint8_t>()[i], fp8_interpretation_t::E4M3);
            out.data_ptr<float>()[i] = halfraw_to_float(halfraw);
        }
        return out;
    }

}

at::Tensor quant_dequant_fp8_e4m3(at::Tensor inputs) {
    return quant_dequant_fp8_e4m3_cuda(inputs.contiguous());
}
