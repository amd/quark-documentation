#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Dict, Any, Union
import torch
import torch.nn as nn
from quark.torch.quantization.tensor_quantize import FakeQuantize
from quark.torch.export.config.config import JsonExporterConfig


class NativeModelInfoBuilder:

    def __init__(self, model: nn.Module, model_type: str, model_dtype: torch.dtype, config: JsonExporterConfig) -> None:
        self.model = model
        self.decoder_type = model_type
        self.model_dtype = model_dtype
        self.config = config
        self.vocab_size = 0
        if hasattr(self.model, "config") and hasattr(self.model.config, "vocab_size"):
            self.vocab_size = self.model.config.vocab_size
        else:
            raise ValueError("Could not find vocab size config in model")

    @staticmethod
    def _contain_quantizer(module: nn.Module) -> bool:
        if ((hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None)
                or (hasattr(module, "_input_quantizer") and module._input_quantizer is not None)
                or (hasattr(module, "_output_quantizer") and module._output_quantizer is not None)):
            return True
        return False

    @staticmethod
    def _build_quant_info(quantizer: FakeQuantize, param_dict: Dict[str, torch.Tensor], tensor_type: str,
                          node_name: str) -> Dict[str, Union[str, int, float, None]]:
        quant_dict: Dict[str, Union[str, int, float, None]] = {}
        scale_name = f"{tensor_type}_scale"
        tensor_name = f"{node_name}.{scale_name}"
        quant_dict["scale"] = tensor_name
        param_dict[tensor_name] = quantizer.scale.detach()

        zero_point_name = f"{tensor_type}_zero_point"
        tensor_name = f"{node_name}.{zero_point_name}"
        quant_dict["zero_point"] = tensor_name
        param_dict[tensor_name] = quantizer.zero_point.detach()

        quant_dict["dtype"] = quantizer.dtype.name if quantizer.dtype is not None else None
        quant_dict["qscheme"] = quantizer.qscheme.name if quantizer.qscheme is not None else None
        quant_dict["ch_axis"] = quantizer.ch_axis
        quant_dict["group_size"] = quantizer.group_size
        quant_dict["round_method"] = quantizer.round_method.name if quantizer.round_method is not None else None
        quant_dict["scale_type"] = quantizer.scale_type.name if quantizer.scale_type is not None else None

        return quant_dict

    @staticmethod
    def _module_to_dict(module: nn.Module, name: str, param_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        module_dict = {}
        if (not list(module.named_children()) or NativeModelInfoBuilder._contain_quantizer(module)):
            if hasattr(module, "weight") and module.weight is not None:
                weight_name = name + ".weight"
                module_dict["weight"] = weight_name
                param_dict[weight_name] = module.weight.detach()

            if hasattr(module, "bias") and module.bias is not None:
                bias_name = name + ".bias"
                module_dict["bias"] = bias_name
                param_dict[bias_name] = module.bias.detach()

            if hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._weight_quantizer, param_dict, "weight",
                                                                      name)
                module_dict["weight_quant"] = quant_info  # type: ignore

            if hasattr(module, "_input_quantizer") and module._input_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._input_quantizer, param_dict, "input",
                                                                      name)
                module_dict["input_quant"] = quant_info  # type: ignore

            if hasattr(module, "_output_quantizer") and module._output_quantizer is not None:
                quant_info = NativeModelInfoBuilder._build_quant_info(module._output_quantizer, param_dict, "output",
                                                                      name)
                module_dict["output_quant"] = quant_info  # type: ignore

            module_dict["type"] = module.__class__.__name__
            return module_dict
        for key, child_module in module.named_children():
            mod_name = name + "." + key
            child_result = NativeModelInfoBuilder._module_to_dict(child_module, mod_name, param_dict)
            module_dict[key] = child_result  # type: ignore
        return module_dict

    def build_model_info(self, param_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        model_dict: Dict[str, Any] = {"config": {}, "structure": {}}

        for name, module in self.model.named_children():
            model_dict["structure"][name] = NativeModelInfoBuilder._module_to_dict(module, name, param_dict)
        model_dict["config"] = self.model.config.to_diff_dict()

        return model_dict

    def build_model_config(self, param_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        model_dict: Dict[str, Any] = {"config": {}, "structure": {}}
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        for name, module in named_modules.items():
            module_dict = {}
            if NativeModelInfoBuilder._contain_quantizer(module):
                if hasattr(module, "weight") and module.weight is not None:
                    weight_name = name + ".weight"
                    module_dict["weight"] = weight_name
                    param_dict[weight_name] = module.weight.detach()

                if hasattr(module, "bias") and module.bias is not None:
                    bias_name = name + ".bias"
                    module_dict["bias"] = bias_name
                    param_dict[bias_name] = module.bias.detach()

                if hasattr(module, "_weight_quantizer") and module._weight_quantizer is not None:
                    module_dict["weight_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._weight_quantizer, param_dict, "weight", name)

                if hasattr(module, "_input_quantizer") and module._input_quantizer is not None:
                    module_dict["input_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._input_quantizer, param_dict, "input", name)

                if hasattr(module, "_output_quantizer") and module._output_quantizer is not None:
                    module_dict["output_quant"] = NativeModelInfoBuilder._build_quant_info(
                        module._output_quantizer, param_dict, "output", name)

                model_dict["structure"][name] = module_dict

        model_dict["config"] = self.model.config.to_diff_dict()

        return model_dict
