#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, cast, TYPE_CHECKING

import torch
import torch.nn as nn
from quark.torch.algorithm.awq.scale import apply_scale
from quark.torch.quantization.config.config import SmoothQuantConfig
if TYPE_CHECKING:
    from quark.torch.algorithm.models.base import \
        BaseQuantModelForCausalLM
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import (get_device, get_named_linears, get_op_name, move_to_device)
from quark.torch.algorithm.utils.prepare import cache_model_inps
from quark.torch.quantization.utils import clear_memory
from tqdm import tqdm
from transformers import PreTrainedModel

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class SmoothQuantProcessor(BaseAlgoProcessor):

    def __init__(self, smooth_model: BaseQuantModelForCausalLM, model: PreTrainedModel,
                 quant_algo_config: SmoothQuantConfig, data_loader: List[Dict[str, torch.Tensor]]) -> None:
        self.smooth_model = smooth_model
        self.model = model
        self.device = model.device
        self.data_loader = data_loader
        self.alpha = quant_algo_config.alpha
        self.clamp_min = quant_algo_config.scale_clamp_min
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.embedding_layers = quant_algo_config.embedding_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.modules, self.module_kwargs, self.inps = self.init_quant()

    def apply(self) -> None:
        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        for i in tqdm(range(len(self.modules)), desc="SmoothQuant"):
            # [STEP 1]: Get layer, extract linear modules, extract act scales
            force_layer_back_to_cpu = False
            if get_device(self.modules[i]) == CPU:
                move_to_device(self.modules[i], self.device)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(self.modules[i])

            named_linears = get_named_linears(self.modules[i])
            input_feat, act_scales = self._get_act_scale_and_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply act scales
            assert self.scaling_layers is not None
            module_config: List[Dict[str, Any]] = self.smooth_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs, self.scaling_layers)
            scales_list = [self._search_best_scale(self.modules[i], act_scales, **layer) for layer in module_config]
            if i != 1:
                continue
            apply_scale(self.modules[i], scales_list, input_feat_dict=None, device=self.device)
            self.modules[i] = cast(
                nn.Module, move_to_device(self.modules[i], CPU if force_layer_back_to_cpu else cur_layer_device))
            clear_memory()

        self.model.config.use_cach = forward_pass_use_cache

    @torch.no_grad()
    def _search_best_scale(self,
                           module: nn.Module,
                           act_scales: Dict[str, torch.Tensor],
                           prev_op: nn.Module,
                           layers: List[nn.Linear],
                           inp: torch.Tensor,
                           module2inspect: Optional[nn.Module] = None,
                           kwargs: Dict[str, Any] = {}) -> Tuple[str, Tuple[str, ...], torch.Tensor]:

        for fc in layers:
            assert isinstance(fc, nn.Linear)
            assert fc.in_features == act_scales[get_op_name(module, fc)].numel()

        # [STEP 1]: Compute maximum of weight

        device, dtype = layers[0].weight.device, layers[0].weight.dtype
        layer_act_scales = act_scales[get_op_name(module, layers[0])].to(device).to(dtype)
        weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in layers], dim=0)
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        # [STEP 2]: Balance quant error bewteen weight and act

        best_scales = (layer_act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)).clamp(min=self.clamp_min)
        assert torch.isnan(best_scales).sum() == 0, best_scales
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)

    @torch.no_grad()
    def _get_act_scale_and_input_feat(
            self, layer: nn.ModuleList,
            named_linears: Dict[str, nn.Linear]) -> Tuple[Dict[str, List[torch.Tensor]], Dict[str, torch.Tensor]]:
        act_scales: Dict[str, torch.Tensor] = {}
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        cur_layer_device = get_device(layer)
        layer_outputs = []
        cache_examples_on_gpu = True

        def stat_tensor(name: str, tensor: torch.Tensor) -> None:
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        # collect act scale
        def cache_input_hook(m: nn.Module, x: Tuple[torch.Tensor, ...], y: torch.Tensor, name: str,
                             feat_dict: Dict[str, List[torch.Tensor]]) -> None:
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name] = []
            stat_tensor(name, x)

        input_feat: Dict[str, List[torch.Tensor]] = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)))

        for j in range(num_batches):
            layer_input = move_to_device(layer_inputs[j], cur_layer_device)
            layer_output = cast(
                torch.Tensor, move_to_device(layer(layer_input)[0], cur_layer_device if cache_examples_on_gpu else CPU))
            layer_outputs.append(layer_output)

        # get output as next layer's input
        self.inps = layer_outputs
        for h in handles:
            h.remove()

        return input_feat, act_scales

    def init_quant(self) -> Tuple[nn.ModuleList, Dict[str, Any], List[torch.Tensor]]:
        assert self.model_decoder_layers is not None
        assert self.embedding_layers is not None
        modules = self.smooth_model.get_model_layers(self.model, self.model_decoder_layers)
        modules, layer_kwargs, inputs = cache_model_inps(self.smooth_model, self.model, modules, self.data_loader,
                                                         self.embedding_layers, self.device)
        return modules, layer_kwargs, inputs
