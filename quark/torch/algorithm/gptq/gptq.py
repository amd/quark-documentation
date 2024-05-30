#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#
# Copyright (c) 2023 潘其威(William)
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, TYPE_CHECKING, Union

import torch
import torch.nn as nn
import transformers
from quark.torch.quantization.config.config import GPTQConfig
if TYPE_CHECKING:
    from quark.torch.algorithm.models.base import \
        BaseQuantModelForCausalLM
from quark.torch.algorithm.processor import BaseAlgoProcessor
from quark.torch.algorithm.utils.module import (get_device, get_named_linears, move_to_device)
from quark.torch.algorithm.utils.prepare import cache_model_inps
from tqdm import tqdm
from transformers import PreTrainedModel

from .quantizer import Quantizer

from quark.torch.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

__all__ = ["GptqProcessor"]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


class GPTQ:

    def __init__(self, layer: nn.Module) -> None:
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H: Optional[torch.Tensor] = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = Quantizer()
        self.inp1: Optional[torch.Tensor] = None
        self.out1: Optional[torch.Tensor] = None

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        assert self.H is not None
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            assert not isinstance(self.layer.padding, str)
            unfold = nn.Unfold(self.layer.kernel_size,
                               dilation=self.layer.dilation,
                               padding=self.layer.padding,
                               stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(self,
                    blocksize: int = 128,
                    percdamp: float = .01,
                    group_size: int = -1,
                    actorder: bool = False,
                    static_groups: bool = False) -> Tuple[torch.Tensor, ...]:
        assert self.H is not None
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = cast(int, perm[idx])
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q

                Losses1[:, i] = (w - q)**2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1

            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f'duration: {(time.time() - tick)}')
        logger.info(f'avg loss: {torch.sum(Losses).item() / self.nsamples}')

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx

    def free(self) -> None:
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        # self.Losses = None
        # self.Trace = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GptqProcessor(BaseAlgoProcessor):

    def __init__(self, gptq_model: BaseQuantModelForCausalLM, model: PreTrainedModel, quant_algo_config: GPTQConfig,
                 data_loader: List[Dict[str, torch.Tensor]]) -> None:
        # assert isinstance(quant_algo_config, GPTQConfig)
        self.gptq_model = gptq_model
        self.model = model
        self.device = model.device
        self.w_bit = quant_algo_config.bit
        self.group_size = quant_algo_config.group_size
        self.sym = quant_algo_config.sym
        # self.version = version
        self.damp_percent = quant_algo_config.damp_percent
        self.act_order = quant_algo_config.desc_act
        self.static_groups = quant_algo_config.static_groups
        self.true_sequential = quant_algo_config.true_sequential
        self.inside_layer_modules = quant_algo_config.inside_layer_modules
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.embedding_layers = quant_algo_config.embedding_layers
        # seqlen = getattr(self.model.config,  self.gptq_model.max_new_tokens_key)
        self.data_loader = data_loader
        self.modules, self.module_kwargs, self.inps = self.init_quant()

    def init_quant(self) -> Tuple[nn.ModuleList, Dict[str, Any], List[torch.Tensor]]:
        assert self.model_decoder_layers is not None
        assert self.embedding_layers is not None
        modules = self.gptq_model.get_model_layers(self.model, self.model_decoder_layers)

        modules, layer_kwargs, inputs = cache_model_inps(self.gptq_model, self.model, modules, self.data_loader,
                                                         self.embedding_layers, self.device)
        return modules, layer_kwargs, inputs

    def apply(self) -> None:
        cache_examples_on_gpu = True
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        layer_outputs: List[torch.Tensor] = []
        forward_pass_use_cache = self.model.config.use_cache
        self.model.config.use_cache = False
        for i in tqdm(range(len(self.modules)), desc="GPTQ"):
            logger.info(f"Start quantizing layer {i + 1}/{len(self.modules)}")
            layer = self.modules[i]
            force_layer_back_to_cpu = False
            if get_device(layer) == CPU:
                move_to_device(layer, self.device)
                force_layer_back_to_cpu = True
            cur_layer_device = get_device(layer)

            full = get_named_linears(layer)
            assert self.inside_layer_modules is not None
            inside_layer_modules: List[str] = self.inside_layer_modules
            if not self.true_sequential:
                inside_layer_modules = [''.join(self.inside_layer_modules)]

            for names in inside_layer_modules:
                subset = {names: full[names]}
                gptq = {}
                for name in subset:
                    gptq[name] = GPTQ(subset[name])
                    gptq[name].quantizer.configure(
                        self.w_bit,
                        perchannel=True,
                        sym=self.sym,
                        mse=False,
                    )

                def add_batch(name: str) -> Callable[[torch.nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:

                    def tmp(_: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
                        gptq[name].add_batch(inp[0].data, out.data)

                    return tmp

                handles = []
                for name in subset:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

                # collect linear input data to calculate Hessian
                for j in range(num_batches):
                    layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                    additional_layer_inputs: Dict[str, Union[None, torch.Tensor, torch.nn.Module]] = {}
                    for k, v in self.module_kwargs.items():
                        if isinstance(v, torch.Tensor):
                            additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                        else:
                            additional_layer_inputs[k] = v

                    additional_layer_inputs['past_key_value'] = None
                    layer(layer_input, **additional_layer_inputs)
                for h in handles:
                    h.remove()

                for name in subset:
                    logger.info(f'Quantizing {name} in layer {i + 1}/{len(self.modules)}...')
                    gptq[name].fasterquant(percdamp=self.damp_percent,
                                           group_size=self.group_size,
                                           actorder=self.act_order,
                                           static_groups=self.static_groups)
                    # subset[name]._weight_quantizer.scale = scale.to(torch.float16)
                    # subset[name]._weight_quantizer.zero_point = zero_point.to(torch.int32)

                    gptq[name].free()

            for j in range(num_batches):
                layer_input = move_to_device(layer_inputs[j], cur_layer_device)
                for k, v in self.module_kwargs.items():
                    if isinstance(v, torch.Tensor):
                        additional_layer_inputs[k] = move_to_device(v, cur_layer_device)
                    else:
                        additional_layer_inputs[k] = v
                additional_layer_inputs['past_key_value'] = None
                layer_output = cast(
                    torch.Tensor,
                    move_to_device(
                        layer(layer_input, **additional_layer_inputs)[0],
                        cur_layer_device if cache_examples_on_gpu else CPU))
                layer_outputs.append(layer_output)

            self.modules[i] = cast(nn.Module, move_to_device(layer,
                                                             CPU if force_layer_back_to_cpu else cur_layer_device))
            del layer
            del gptq
            del layer_inputs
            layer_inputs, layer_outputs = layer_outputs, []
            self.model.config.use_cache = forward_pass_use_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
