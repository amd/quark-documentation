#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import torch
from typing import List, Tuple, Any, Dict, TYPE_CHECKING
import torch.nn as nn
from quark.torch.algorithm.utils.module import get_device, move_to_device
from quark.torch.algorithm.utils.utils import clear_memory
if TYPE_CHECKING:
    from quark.torch.algorithm.models.base import BaseQuantModelForCausalLM
from transformers import PreTrainedModel


def cache_model_inps(quant_model: BaseQuantModelForCausalLM, model: PreTrainedModel, modules: nn.ModuleList,
                     samples: List[Dict[str, Any]], embedding_layer_name_list: List[str],
                     device: torch.device) -> Tuple[nn.ModuleList, Dict[str, Any], List[torch.Tensor]]:

    inps: List[torch.Tensor] = []
    layer_kwargs: Dict[str, Any] = {}

    modules[0] = modules[0].to(device)
    quant_model.move_embed(model, embedding_layer_name_list, device)

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):

        def __init__(self, module: nn.Module) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args: torch.Tensor, **kwargs: Any) -> None:
            # assume first input to forward is hidden states
            if len(args) > 0:
                hidden_states = args[0]
                del args
            else:
                first_key = list(kwargs.keys())[0]
                hidden_states = kwargs.pop(first_key)

            inps.append(hidden_states)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs

    cur_layer_device = get_device(modules[0])
    modules[0] = Catcher(modules[0])
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            try:
                model(sample)
            except ValueError:  # work with early exit
                pass
        else:
            for k, v in sample.items():
                if len(v.shape) == 1:
                    v = v.unsqueeze(0)
                sample[k] = move_to_device(v, cur_layer_device)
            try:
                model(**sample)
            except ValueError:  # work with early exit
                pass
    del samples
    modules[0] = modules[0].module  # restore
    # inps = inps[0]

    modules[0] = modules[0].cpu()
    quant_model.move_embed(model, embedding_layer_name_list, torch.device("cpu"))

    clear_memory()

    if layer_kwargs.get("attention_mask") is not None:
        layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(device)

    return modules, layer_kwargs, inps
