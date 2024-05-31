#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import Any, Tuple, Union
from typing import Dict, Type, cast, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from quark.torch.algorithm.awq import AwqProcessor
from quark.torch.algorithm.awq.smooth import SmoothQuantProcessor
from quark.torch.quantization.config.config import AlgoConfig
from quark.torch.algorithm.gptq import GptqProcessor
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import get_nested_attr_from_module

from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast


class BaseQuantModelForCausalLM(nn.Module):

    def __init__(self, model: PreTrainedModel, is_quantized: bool, config: PretrainedConfig,
                 algo_config: AlgoConfig) -> None:
        super().__init__()
        self.model = model
        self.is_quantized = is_quantized
        self.config = config
        self.algo_config = algo_config

    # def to(self, device: Union[str, torch.device]) -> nn.Module:
    #     return cast(nn.Module, self.model.to(device))

    def forward(self, *args: Any, **kwargs: Any) -> Union[Tuple[Any], CausalLMOutputWithPast]:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Union[GenerateOutput, torch.LongTensor]:
        with torch.inference_mode():
            return self.model.generate(*args, **kwargs)

    @torch.no_grad()
    def quantize(
        self, data_loader: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                                 DataLoader[Dict[str, torch.Tensor]]]
    ) -> None:

        quantizer_cls_map: Dict[str, Type[BaseAlgoProcessor]] = {
            "awq": AwqProcessor,
            "gptq": GptqProcessor,
            "smoothquant": SmoothQuantProcessor
        }

        if (quantizer_cls := quantizer_cls_map.get(self.algo_config.name, None)) is None:
            raise NotImplementedError(f"{self.algo_config.name} is not supported.")

        quantizer = quantizer_cls(self, self.model, self.algo_config, data_loader)
        quantizer.apply()
        self.is_quantized = True

    @staticmethod
    def get_model_layers(model: nn.Module, decoder_layers_name: str) -> nn.ModuleList:
        decoder_layers = get_nested_attr_from_module(model, decoder_layers_name)
        return cast(nn.ModuleList, decoder_layers)

    @staticmethod
    def move_embed(model: nn.Module, embedding_layer_name_list: List[str], device: torch.device) -> None:
        for embedding_layer_name in embedding_layer_name_list:
            embedding_layer = get_nested_attr_from_module(model, embedding_layer_name)
            embedding_layer = embedding_layer.to(device)

    @staticmethod
    def get_layers_for_scaling(module: nn.Module, input_feat: Dict[str, Any], module_kwargs: Dict[str, Any],
                               scaling_layers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []

        for layer in scaling_layers:

            if "condition" in layer:
                condition_result = eval(layer["condition"])
                if not condition_result:
                    continue

            linear_layers = []
            for i in range(len(layer['layers'])):
                linear_layers.append(get_nested_attr_from_module(module, layer['layers'][i]))

            leyer_dict = dict(
                prev_op=get_nested_attr_from_module(module, layer['prev_op']),
                layers=linear_layers,
                inp=input_feat[layer['inp']],
            )

            if layer['module2inspect'] is not None:
                leyer_dict['module2inspect'] = get_nested_attr_from_module(module, layer['module2inspect'])
            if layer['has_kwargs'] is True:
                leyer_dict['kwargs'] = module_kwargs

            layers.append(leyer_dict)

        return layers
