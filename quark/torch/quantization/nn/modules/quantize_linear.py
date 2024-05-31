#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Dict, Any, Optional
import torch
from torch import nn
from torch.nn import functional as F
from .mixin import QuantMixin
from quark.torch.quantization.config.config import QuantizationConfig

__all__ = ["QuantLinear"]


class QuantLinear(nn.Linear, QuantMixin):
    """Quantized version of nn.Linear

    """

    def __init__(self, in_features: int, out_features: int, bias: bool, quant_config: QuantizationConfig,
                 **kwargs: Any) -> None:
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        input_qspec = quant_config.input_tensors
        output_qspec = quant_config.output_tensors
        weight_qspec = quant_config.weight
        self.init_quantizer(input_qspec, output_qspec, weight_qspec, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        quant_input = self._input_quantizer(input) if self._input_quantizer else input
        quant_weight = self._weight_quantizer(self.weight) if self._weight_quantizer else self.weight

        output = F.linear(quant_input, quant_weight, bias=self.bias)
        quant_output: torch.Tensor = self._output_quantizer(output) if self._output_quantizer else output

        return quant_output

    @classmethod
    def from_float(cls,
                   float_module: nn.Module,
                   layer_quant_config: QuantizationConfig,
                   reload: bool = False,
                   weight_tensor: Optional[torch.Tensor] = None,
                   bias_tensor: Optional[torch.Tensor] = None) -> nn.Linear:
        bias = False if (float_module.bias is None) and (reload is False or bias_tensor is None) else True
        quant_linear = cls(float_module.in_features, float_module.out_features, bias, layer_quant_config, reload=reload)
        if reload is True and weight_tensor is not None:
            quant_linear.weight.data = weight_tensor.to(float_module.weight.device)
        else:
            quant_linear.weight = float_module.weight

        if reload is True and bias_tensor is not None:
            quant_linear.bias.data = bias_tensor.to(float_module.weight.device)
        else:
            quant_linear.bias = float_module.bias
        return quant_linear

    def load_quant_params(self, params_dict: Dict[str, torch.Tensor]) -> None:
        device = self.weight.device
        if hasattr(self, "_input_quantizer") and self._input_quantizer is not None:
            if params_dict.get("input_scale", None) is not None:
                self._input_quantizer.scale.data = params_dict["input_scale"].to(device)
            if params_dict.get("input_zero_point", None) is not None:
                self._input_quantizer.zero_point.data = params_dict["input_zero_point"].to(device)

        if hasattr(self, "_output_quantizer") and self._output_quantizer is not None:
            if params_dict.get("output_scale", None) is not None:
                self._output_quantizer.scale.data = params_dict["output_scale"].to(device)
            if params_dict.get("output_zero_point", None) is not None:
                self._output_quantizer.zero_point.data = params_dict["output_zero_point"].to(device)

        if hasattr(self, "_weight_quantizer") and self._weight_quantizer is not None:
            if params_dict.get("weight_scale", None) is not None:
                self._weight_quantizer.scale.data = params_dict["weight_scale"].to(device)
            if params_dict.get("weight_zero_point", None) is not None:
                self._weight_quantizer.zero_point.data = params_dict["weight_zero_point"].to(device)
