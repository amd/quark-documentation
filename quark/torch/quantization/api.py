#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for PyTorch."""

import fnmatch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple, Iterable
from dataclasses import fields
from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.config_verification import init_quantization_config, verify_quantization_spec
from quark.torch.quantization.transformation.model_transformation import set_op_by_name, get_op_by_name
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.tensor_quantize import FakeQuantize, FreezedFakeQuantize
from quark.torch.utils.log import DebugLogger, ScreenLogger

__all__ = ["ModelQuantizer"]

in_place_replace_ops = DebugLogger(name="in_place_replace_ops")
logger = ScreenLogger(__name__)


class ModelQuantizer:
    """
    Provides an API for quantizing deep learning models using PyTorch. This class handles the configuration and processing of the model for quantization based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary quantization parameters defined. This class assumes that the model is compatible with the quantization settings specified in 'config'.

    Args:
        config (Config): Configuration object containing settings for quantization.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.is_all_dynamic: Optional[bool] = None
        self.is_weight_only: Optional[bool] = None
        self.model: Optional[nn.Module] = None
        self.init_config()
        self.device = None

    def quantize_model(
        self, model: nn.Module, dataloader: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                                  DataLoader[Dict[str, torch.Tensor]]]
    ) -> nn.Module:
        """
        This function aims to quantize the given PyTorch model to optimize its performance and reduce its size. This function accepts a model and a torch dataloader. The dataloader is used to provide data necessary for calibration during the quantization process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the quantization approach accordingly.It's important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the quantization process.

        Parameters:
            model (nn.Module): The PyTorch model to be quantized. This model should be already trained and ready for quantization.
            dataloader (Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]):
                The DataLoader providing data that the quantization process will use for calibration. This can be a simple DataLoader returning
                tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

        Returns:
            nn.Module: The quantized version of the input model. This model is now optimized for inference with reduced size and potentially improved
            performance on targeted devices.

        **Examples**:

            .. code-block:: python

                # Model & Data preparation
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
                model.eval()
                tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
                from quark.torch.quantization.config.config import Config
                from quark.torch.quantization.config.custom_config import DEFAULT_W_UINT4_PER_GROUP_CONFIG
                quant_config = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)
                from torch.utils.data import DataLoader
                text = "Hello, how are you?"
                tokenized_outputs = tokenizer(text, return_tensors="pt")
                calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

                from quark.torch import ModelQuantizer
                quantizer = ModelQuantizer(quant_config)
                quant_model = quantizer.quantize_model(model, calib_dataloader)

        """

        self.model = model

        # step1[optional]: pre quant optimization

        # step2: prepare quantizable model, in-place replace linear or other types which need quantization
        # setup  quant config per layer
        logger.info("In-place OPs replacement start.")
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        module_configs: Dict[str, QuantizationConfig] = {}
        for name, module in named_modules.items():
            if isinstance(module, torch.nn.Linear):
                excluded = False
                for name_pattern in self.config.exclude:
                    if fnmatch.fnmatch(name, name_pattern):
                        excluded = True
                        break
                if excluded:
                    continue

                reset = False
                for name_pattern, quant_config in self.config.layer_quant_config.items():
                    if fnmatch.fnmatch(name, name_pattern):
                        module_configs[name] = quant_config
                        reset = True
                        break

                if not reset:
                    module_configs[name] = self.config.global_quant_config
        # Inplace replace
        for name, module in tqdm(named_modules.items()):
            if name in module_configs:
                qlinear_module = QuantLinear.from_float(module, module_configs[name])
                set_op_by_name(self.model, name, qlinear_module)
                in_place_replace_ops.debug(name)
        logger.info("In-place OPs replacement end.")

        # step3[optional]: apply advanced quant algo such as gptq/awq ...
        for module in self.model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_fake_quant()
                module.disable_observer()

        if self.config.algo_config is not None:
            self.apply_advanced_quant_algo(dataloader)

        # step4[optional]: do calibration
        # just calib, turn off quantize
        if self.is_all_dynamic is True:
            logger.info("Dynamic quantization, no calibration.")
        elif self.is_weight_only is True:
            logger.info("Weight only quantization start.")
            for module in self.model.modules():
                if isinstance(module, FakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            # TODO: Change SmoothQuant to Pre-Process
            if (self.config.algo_config is None) or self.config.algo_config.name in ['smoothquant']:
                for data in dataloader:
                    if isinstance(data, dict):
                        self.model(**data)
                    else:
                        self.model(data)
                    break
            logger.info("Weight only quantization end.")
        else:
            logger.info("Calibration start.")
            for module in self.model.modules():
                if isinstance(module, FakeQuantize):
                    module.enable_observer()
                    module.disable_fake_quant()

            for data in tqdm(dataloader):
                if isinstance(data, dict):
                    self.model(**data)
                else:
                    self.model(data)
            logger.info("Calibration end.")
        logger.info("Model quantization has been completed.")

        # step5[optional]: do evaluation, turn on quantize
        # TODO: Add fake_quant to GPTQ
        if (self.config.algo_config) and self.config.algo_config.name in ['gptq']:
            for module in self.model.modules():
                if isinstance(module, FakeQuantize):
                    module.disable_observer()
                    module.disable_fake_quant()
        else:
            for module in self.model.modules():
                if isinstance(module, FakeQuantize):
                    module.disable_observer()
                    module.enable_fake_quant()
        return self.model

    def freeze(self, model: nn.Module) -> nn.Module:
        """
        Freezes the quantized model by replacing FakeQuantize modules with FreezedFakeQuantize modules.
        If Users want to export quantized model to torch_compile, please freeze model first.

        Args:
            model (nn.Module): The neural network model containing quantized layers.

        Returns:
            nn.Module: The modified model with FakeQuantize modules replaced by FreezedFakeQuantize modules.
        """
        logger.info("Freeze model start.")
        assert isinstance(self.model, nn.Module)
        named_modules = dict(self.model.named_modules(remove_duplicate=False))
        for name, module in named_modules.items():
            if isinstance(module, FakeQuantize):
                freezed_qlinear_module = FreezedFakeQuantize.from_fake_quantize(module)
                set_op_by_name(self.model, name, freezed_qlinear_module)
        logger.info("Freeze model end.")
        return model

    def init_config(self) -> None:
        logger.info("Configuration checking start.")
        config = self.config
        verify_quantization_spec(config)
        # TODO: Verify quant algo

        self.is_all_dynamic = True
        self.is_weight_only = True
        for field in fields(Config):
            if field.name in ["global_quant_config"]:
                quantization_config = getattr(config, field.name)
                is_dynamic, is_weight_only = init_quantization_config(quantization_config)
                if is_weight_only is False:
                    self.is_weight_only = is_weight_only
                if is_dynamic is False:
                    self.is_all_dynamic = False
            elif field.name in ["layer_type_quant_config", "layer_quant_config"]:
                quantization_config_list = getattr(config, field.name)
                for quantization_config in quantization_config_list.values():
                    is_dynamic, is_weight_only = init_quantization_config(quantization_config)
                    if is_weight_only is False:
                        self.is_weight_only = is_weight_only
                    if is_dynamic is False:
                        self.is_all_dynamic = False

        config_parsing_result = ''
        if self.is_weight_only:
            config_parsing_result = 'weight only'
        elif self.is_all_dynamic:
            config_parsing_result = 'dynamic'
        else:
            config_parsing_result = 'static'
        logger.info(
            f"Configuration checking end. The configuration is effective. This is {config_parsing_result} quantization."
        )

    def apply_advanced_quant_algo(
        self, dataloader: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                DataLoader[Dict[str, torch.Tensor]]]
    ) -> None:
        logger.info("Advanced algorithm start.")
        assert self.config.algo_config is not None
        assert self.model is not None
        self.device = self.model.device

        from quark.torch.algorithm.models.base import BaseQuantModelForCausalLM

        self.wraped_model_with_algo = BaseQuantModelForCausalLM(model=self.model,
                                                                is_quantized=False,
                                                                config=self.model.config,
                                                                algo_config=self.config.algo_config)

        self.wraped_model_with_algo.quantize(dataloader)

        self.model = self.wraped_model_with_algo.model.cpu()
        assert self.model is not None
        self.model = self.model.to(self.device)
        logger.info("Advanced algorithm end.")


def get_name_and_info(model_info: Dict[str, Any], parent_key: str = "") -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, value in model_info.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            if value.get("type", None) is not None and value.get("weight", None) is not None:
                yield new_key, value
            else:
                yield from get_name_and_info(value, new_key)
        else:
            continue


def from_float_and_dict(float_module: nn.Module, quant_info: Dict[str, Any],
                        param_dict: Dict[str, torch.Tensor]) -> nn.Module:
    input_tensors = None
    quant_params: Dict[str, Optional[torch.Tensor]] = {}
    if quant_info.get("input_quant", None) is not None:
        input_tensors = QuantizationSpec.from_dict(quant_info["input_quant"])
        scale_name = quant_info["input_quant"]["scale"]
        quant_params["input_scale"] = param_dict[scale_name]
        zero_point_name = quant_info["input_quant"]["zero_point"]
        quant_params["input_zero_point"] = param_dict[zero_point_name]

    output_tensors = None
    if quant_info.get("output_quant", None) is not None:
        output_tensors = QuantizationSpec.from_dict(quant_info["output_quant"])
        scale_name = quant_info["output_quant"]["scale"]
        quant_params["output_scale"] = param_dict[scale_name]
        zero_point_name = quant_info["output_quant"]["zero_point"]
        quant_params["output_zero_point"] = param_dict[zero_point_name]

    weight = None
    if quant_info.get("weight_quant", None) is not None:
        weight = QuantizationSpec.from_dict(quant_info["weight_quant"])
        scale_name = quant_info["weight_quant"]["scale"]
        quant_params["weight_scale"] = param_dict[scale_name]
        zero_point_name = quant_info["weight_quant"]["zero_point"]
        quant_params["weight_zero_point"] = param_dict[zero_point_name]

    module_config = QuantizationConfig(input_tensors=input_tensors, output_tensors=output_tensors, weight=weight)

    weight_tensor = param_dict[quant_info.get("weight", None)]
    bias_tensor = None
    if quant_info.get("bias", None) is not None:
        bias_tensor = param_dict[quant_info.get("bias", None)]

    quant_module = QuantLinear.from_float(float_module,
                                          module_config,
                                          reload=True,
                                          weight_tensor=weight_tensor,
                                          bias_tensor=bias_tensor)
    quant_module.load_quant_params(quant_params)
    return quant_module


def from_exported_files(model: nn.Module, json_path: str, safetensors_path: str) -> nn.Module:
    import json
    from safetensors.torch import load_file
    # load model structure and parameters
    with open(json_path, "r") as file:
        model_dict = json.load(file)
    params_dict = load_file(safetensors_path)

    # verify exported model and float model have the same configuration
    model_config = model_dict["config"]
    float_model_config = model.config.to_diff_dict()
    if (json.dumps(model_config) != json.dumps(float_model_config)):
        logger.error("Exported model and float model are not the same model!")
    # assert ((json.dumps(model_config) == json.dumps(float_model_config)),
    #         "Exported model and float model are not the same model!")

    logger.info("In-place OPs replacement start.")
    for name, module_info in get_name_and_info(model_dict["structure"]):
        float_module = get_op_by_name(model, name)
        if module_info["type"] == "QuantLinear":
            module = from_float_and_dict(float_module, module_info, params_dict)
            set_op_by_name(model, name, module)
        else:
            device = float_module.weight.device
            float_module.weight.data = params_dict[module_info.get("weight", None)].to(device)

        # print(name, module_info)

    logger.info("In-place OPs replacement end.")
    return model
