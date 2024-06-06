#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# type: ignore

import torch

from .extensions import kernel_ext

from torch.library import Library, impl, impl_abstract
from typing import Any

# namespace
quant_scope_lib = Library("quant_scope", "DEF")

quant_scope_lib.define("quant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e4m3", "CompositeExplicitAutograd")
def quant_fp8_e4m3(inputs: torch.Tensor) -> Any:
    inputs = torch.clamp(inputs, min=-448, max=448)
    return inputs.to(torch.float8_e4m3fn)


quant_scope_lib.define("dequant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e4m3", "CompositeExplicitAutograd")
def dequant_fp8_e4m3(inputs: torch.Tensor) -> Any:
    return inputs.to(torch.float16)


quant_scope_lib.define("quant_fp8_e4m3_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e4m3_with_scale", "CompositeExplicitAutograd")
def quant_fp8_e4m3_with_scale(inputs: torch.Tensor, scale: float) -> Any:
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-448, max=448)
    return inputs.to(torch.float8_e4m3fn)


quant_scope_lib.define("dequant_fp8_e4m3_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e4m3_with_scale", "CompositeExplicitAutograd")
def dequant_fp8_e4m3_with_scale(inputs: torch.Tensor, scale: float) -> Any:
    return inputs.to(torch.float16) * scale


quant_scope_lib.define("quant_dequant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_dequant_fp8_e4m3", "CompositeExplicitAutograd")
def quant_dequant_fp8_e4m3(inputs: torch.Tensor) -> Any:
    inputs_type = inputs.dtype
    inputs = torch.clamp(inputs, min=-448, max=448)
    outputs = (inputs).to(torch.float8_e4m3fn).to(inputs_type)
    return outputs


quant_scope_lib.define("quant_dequant_fp8_e4m3_with_scale(Tensor x, Tensor scale) -> Tensor")


@impl(quant_scope_lib, "quant_dequant_fp8_e4m3_with_scale", "CompositeExplicitAutograd")
def quant_dequant_fp8_e4m3_with_scale(inputs: torch.Tensor, scale: torch.Tensor) -> Any:
    inputs_type = inputs.dtype
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-448, max=448)
    return inputs.to(torch.float8_e4m3fn).to(inputs_type) * scale


@impl_abstract("quant_scope::quant_dequant_fp8_e4m3_with_scale")
def _(inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(inputs)


quant_scope_lib.define(
    "fake_quantize_per_tensor_affine(Tensor inputs, Tensor scale, Tensor zero_point, int quant_min, int quant_max, int round_mode) -> Tensor"
)


@impl(quant_scope_lib, "fake_quantize_per_tensor_affine", "CompositeExplicitAutograd")
def fake_quantize_per_tensor_affine(inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int,
                                    quant_max: int, round_mode: int) -> Any:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)
    if kernel_ext is not None and inputs.device != torch.device('cpu'):
        res = kernel_ext.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max, round_mode)
    else:
        res = torch.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max)
    if inputs_type != res.dtype:
        res = res.to(inputs_type)
    if scale_type != scale.dtype:
        scale = scale.to(scale_type)
    return res


@impl_abstract("quant_scope::fake_quantize_per_tensor_affine")
def _(inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int,
      round_mode: int) -> torch.Tensor:
    return torch.empty_like(inputs)


quant_scope_lib.define(
    "fake_quantize_per_channel_affine(Tensor inputs, Tensor scale, Tensor zero_point, int axis,  int quant_min, int quant_max, int round_mode) -> Tensor"
)


@impl(quant_scope_lib, "fake_quantize_per_channel_affine", "CompositeExplicitAutograd")
def fake_quantize_per_channel_affine(inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int,
                                     quant_min: int, quant_max: int, round_mode: int) -> Any:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)
    res = torch.fake_quantize_per_channel_affine(inputs, scale, zero_point, axis, quant_min, quant_max)
    if inputs_type != res.dtype:
        res = res.to(inputs_type)
    if scale_type != scale.dtype:
        scale = scale.to(scale_type)
    return res


@impl_abstract("quant_scope::fake_quantize_per_channel_affine")
def _(inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int, quant_max: int,
      round_mode: int) -> torch.Tensor:
    return torch.empty_like(inputs)
