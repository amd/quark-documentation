#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# type: ignore

import os
import torch
from torch.library import Library, impl
from types import ModuleType
from typing import Any, List, Optional
from torch import ops  # type: ignore[attr-defined]
from .hw_emulation import hw_emulation_interface
from torch.autograd import Function
from typing import Any, Union
from torch.onnx._internal import jit_utils
from torch.onnx import errors, symbolic_helper
import torch._C._onnx as _C_onnx


class RoundMode():
    FLOOR = 2
    ROUND = 3
    NEARBYINT = 8


class QuantE4M3Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quant_scope.quant_fp8_e4m3(inputs)
        else:
            return ops.quant_scope.quant_fp8_e4m3_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore
        return grad_outputs


class DequantE4M3Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quant_scope.dequant_fp8_e4m3(inputs)
        else:
            return ops.quant_scope.dequant_fp8_e4m3_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore
        return grad_outputs


class QuantDequantE4M3Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:  # type: ignore
        # if scale is None:
        #     return ops.quant_scope.quant_dequant_fp8_e4m3(inputs)
        return ops.quant_scope.quant_dequant_fp8_e4m3_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore
        return grad_outputs, None

    @staticmethod
    @symbolic_helper.parse_args("v", "v")
    def symbolic(g: jit_utils.GraphContext,
                 inputs: torch.Tensor,
                 scale: Union[torch.Tensor, None] = None) -> torch.Value:
        if scale is None:
            scale = torch.tensor(1.0)
        zero_point = torch.tensor(0, dtype=torch.float8_e4m3fn)
        quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
        return g.op("DequantizeLinear", quantized, scale, zero_point)


class FakeQuantizePerTensorAffine(Function):

    @staticmethod
    def forward(  # type: ignore
            ctx: Any, inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int,
            quant_max: int, round_mode: int) -> Any:
        outputs = ops.quant_scope.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max,
                                                                  round_mode)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore
        return grad_outputs, None, None, None, None, None

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
    def symbolic(g: jit_utils.GraphContext, inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                 quant_min: int, quant_max: int, round_mode: int) -> torch.Value:
        if (quant_min, quant_max) not in [(0, 255), (-128, 127), (-8, 7), (0, 15)]:
            raise errors.SymbolicValueError(
                "For (quant_min, quant_max), ONNX allows only (0, 255), (-128, 127), (-8, 7) and (0, 15). "
                f"Got ({quant_min}, {quant_max})", )

        #zero_point = torch.tensor(zero_point)
        if quant_min == 0:
            zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
        else:
            zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
        quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
        return g.op("DequantizeLinear", quantized, scale, zero_point)


class FakeQuantizePerChannelAffine(Function):

    @staticmethod
    def forward(  # type: ignore
            ctx: Any, inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int,
            quant_max: int, round_mode: int) -> Any:
        outputs = ops.quant_scope.fake_quantize_per_channel_affine(inputs, scale, zero_point, axis, quant_min,
                                                                   quant_max, round_mode)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore
        return grad_outputs, None, None, None, None, None, None

    @staticmethod
    @symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "i")
    def symbolic(g: jit_utils.GraphContext, inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                 axis: int, quant_min: int, quant_max: int, round_mode: int) -> torch.Value:
        if (quant_min, quant_max) not in [(0, 255), (-128, 127), (-8, 7), (0, 15)]:
            raise errors.SymbolicValueError(
                "For (quant_min, quant_max), ONNX allows only (0, 255), (-128, 127), (-8, 7) and (0, 15). "
                f"Got ({quant_min}, {quant_max})", )

        #zero_point = torch.tensor(zero_point)
        if quant_min == 0:
            zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
        else:
            zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
        quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
        return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)


fake_quantize_per_tensor_affine = FakeQuantizePerTensorAffine.apply
quant_fp8_e4m3 = QuantE4M3Function.apply
dequant_fp8_e4m3 = DequantE4M3Function.apply
quant_dequant_fp8_e4m3 = QuantDequantE4M3Function.apply
fake_quantize_per_channel_affine = FakeQuantizePerChannelAffine.apply
