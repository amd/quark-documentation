#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, cast, TYPE_CHECKING
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from quark.torch.algorithm.utils.utils import clear_memory
from quark.torch.algorithm.utils.module import get_op_name, append_str_prefix, get_named_linears
from quark.torch.algorithm.awq.scale import apply_clip, apply_scale
from quark.torch.algorithm.utils.prepare import cache_model_inps
import functools
from collections import defaultdict
from transformers import PreTrainedModel
if TYPE_CHECKING:
    from quark.torch.algorithm.models.base import BaseQuantModelForCausalLM
from quark.torch.algorithm.processor import BaseAlgoProcessor

__all__ = ["AwqProcessor"]


class AwqProcessor(BaseAlgoProcessor):

    def __init__(self, awq_model: BaseQuantModelForCausalLM, model: PreTrainedModel, quant_algo_config: Any,
                 data_loader: DataLoader[List[Dict[str, torch.Tensor]]]) -> None:
        # assert isinstance(quant_algo_config, AWQConfig)
        self.awq_model = awq_model
        self.model = model
        self.device = model.device
        self.w_bit = quant_algo_config.bit
        self.group_size = quant_algo_config.group_size
        self.sym = quant_algo_config.sym
        # self.version = version
        self.data_loader = data_loader
        self.embedding_layers = quant_algo_config.embedding_layers
        self.model_decoder_layers = quant_algo_config.model_decoder_layers
        self.scaling_layers = quant_algo_config.scaling_layers
        self.modules, self.module_kwargs, self.inps = self.init_quant()

        # self.fake_quantize = self.init_fake_quantize()

    def apply(self) -> None:
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            # Move module and inputs to correct device
            common_device = next(self.modules[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                self.modules[i] = self.modules[i].to(self.device)
                common_device = next(self.modules[i].parameters()).device

            self.inps = self.inps.to(common_device)

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])
            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict[str,
                                     Any]] = self.awq_model.get_layers_for_scaling(self.modules[i], input_feat,
                                                                                   self.module_kwargs,
                                                                                   self.scaling_layers)
            scales_list = [self._search_best_scale(self.modules[i], **layer) for layer in module_config]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat, device=self.device)
            scales_list = append_str_prefix(scales_list, get_op_name(self.model, self.modules[i]) + ".")

            # [STEP 3]: Compute and apply clipping list
            clip_list = self._search_best_clip(named_linears, input_feat)
            apply_clip(self.modules[i], clip_list, self.device)
            clip_list = append_str_prefix(clip_list, get_op_name(self.model, self.modules[i]) + ".")
            # clear_memory()

            # [STEP 4]: Quantize weights
            self._apply_quant(named_linears)

    @torch.no_grad()
    def _search_best_scale(self,
                           module: nn.Module,
                           prev_op: nn.Module,
                           layers: List[nn.Linear],
                           inp: torch.Tensor,
                           module2inspect: Optional[nn.Module] = None,
                           kwargs: Dict[str, Any] = {}) -> Tuple[str, Tuple[str, ...], torch.Tensor]:
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        if "past_key_value" in kwargs:
            kwargs.pop("past_key_value")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute maximum of weight
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        weight = weight.view(-1, self.group_size)
        w_scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
        w_scale = w_scale.view(org_shape)
        w_max = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute maximum of x
        x_max = inp.abs().view(-1, inp.shape[-1]).mean(0)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            fp16_output = module2inspect(inp, **kwargs)
            if isinstance(fp16_output, tuple):
                fp16_output = fp16_output[0]

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(inp, w_max, x_max, module2inspect, layers, fp16_output, kwargs)

        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)

    def _compute_best_scale(self,
                            x: torch.Tensor,
                            w_max: torch.Tensor,
                            x_max: torch.Tensor,
                            module2inspect: nn.Module,
                            linears2scale: List[nn.Linear],
                            fp16_output: torch.Tensor,
                            kwargs: Dict[str, Any] = {}) -> torch.Tensor:

        n_grid = 20
        best_ratio = -1.0
        best_scales = None
        best_error = float('inf')

        org_sd = {k: v.cpu() for k, v in module2inspect.state_dict().items()}

        device = x.device
        x_max = x_max.view(-1).to(device)
        w_max = w_max.view(-1).to(device)

        for i in range(n_grid):
            # create new scales
            ratio = i / n_grid
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales_view = scales.view(1, -1).to(device)

            # Q(W * s)
            for fc in linears2scale:
                fc.weight.mul_(scales_view)
                fc.weight.data = self.pseudo_quantize_tensor(fc.weight.data, fc) / scales_view

            # W * X

            int_w_output = module2inspect(x, **kwargs)
            if isinstance(int_w_output, tuple):
                int_w_output = int_w_output[0]

            # compute mean squared error (L2 norm)
            loss = (fp16_output - int_w_output).float().pow(2).mean().item()  # NOTE: float prevents overflow

            # history.append(loss)
            if loss < best_error:
                best_error = loss
                best_ratio = ratio
                best_scales = scales.clone()
            module2inspect.load_state_dict(org_sd)

        if best_ratio == -1.0:
            raise Exception

        assert best_scales is not None
        assert torch.isnan(best_scales).sum() == 0

        return best_scales.detach().cpu()

    @torch.no_grad()
    def _search_best_clip(self, named_linears: Dict[str, nn.Linear],
                          input_feat: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
        clip_list: List[Tuple[str, torch.Tensor]] = []
        avoid_clipping = ["q_", "k_", "query", "key", "Wqkv"]

        for name in named_linears:
            # due to qk bmm, it is hard to clip precisely
            if any([_ in name for _ in avoid_clipping]):
                continue

            named_linears[name].to(self.device)
            max_val = self._compute_best_clip(named_linears[name], input_feat[name])
            clip_list.append((name, max_val))

            named_linears[name].cpu()

        return clip_list

    @torch.no_grad()
    def _compute_best_clip(self,
                           named_linears: torch.nn.Linear,
                           input_feat: torch.Tensor,
                           n_grid: int = 20,
                           max_shrink: float = 0.5,
                           n_sample_token: int = 512) -> torch.Tensor:
        w = named_linears.weight
        assert w.dim() == 2
        # w           [co, ci]      -> [co, 1, n_group, group size]
        # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
        group_size = self.group_size if self.group_size > 0 else w.shape[1]
        input_feat = input_feat.view(-1, input_feat.shape[-1])
        input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
        n_sample_token = min(input_feat.shape[1], n_sample_token)
        input_feat = input_feat[:, 0::input_feat.shape[1] // n_sample_token]
        w = w.reshape(w.shape[0], 1, -1, group_size)

        oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
        assert w.shape[0] % oc_batch_size == 0
        w_all = w
        best_max_val_all = []

        for i_b in range(w.shape[0] // oc_batch_size):
            w = w_all[i_b * oc_batch_size:(i_b + 1) * oc_batch_size]

            org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

            best_max_val = org_max_val.clone()
            min_errs = torch.ones_like(org_max_val) * 1e9
            input_feat = input_feat.to(w.device)
            org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

            for i_s in range(int(max_shrink * n_grid)):
                max_val = org_max_val * (1 - i_s / n_grid)
                min_val = -max_val
                cur_w = torch.clamp(w, min_val, max_val)
                q_w = cur_w
                q_w = self.pseudo_quantize_tensor(cur_w, named_linears)
                cur_out = (input_feat * q_w).sum(dim=-1)

                # co, 1, n_group, 1
                err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
                del cur_w
                del cur_out
                cur_best_idx = err < min_errs
                min_errs[cur_best_idx] = err[cur_best_idx]
                best_max_val[cur_best_idx] = max_val[cur_best_idx]
            best_max_val_all.append(best_max_val)

        best_max_val = torch.cat(best_max_val_all, dim=0)

        clear_memory(input_feat)
        clear_memory(org_out)

        return best_max_val.squeeze(1)

    def _get_input_feat(self, layer: nn.Module, named_linears: Dict[str, nn.Linear]) -> Dict[str, torch.Tensor]:
        # firstly, get input features of all linear layers
        def cache_input_hook(m: nn.Module, x: Tuple[torch.Tensor], y: torch.Tensor, name: str,
                             feat_dict: Dict[str, List[torch.Tensor]]) -> None:
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat: Dict[str, List[torch.Tensor]] = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(named_linears[name].register_forward_hook(
                functools.partial(cache_input_hook, name=name, feat_dict=input_feat)))
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input
        self.inps = layer(self.inps, **self.module_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    @torch.no_grad()
    def pseudo_quantize_tensor(self,
                               w: torch.Tensor,
                               linear_layer: nn.Linear,
                               get_scale_zp: bool = False) -> torch.Tensor:
        from quark.torch.quantization.tensor_quantize import FakeQuantize
        # org_w_shape = w.shape
        # if self.group_size > 0:
        #     assert org_w_shape[-1] % self.group_size == 0
        #     w = w.reshape(-1, self.group_size)
        # assert w.dim() == 2

        for module in linear_layer.modules():
            if isinstance(module, FakeQuantize):
                module.enable_observer()
                module.enable_fake_quant()
        w_q = linear_layer._weight_quantizer(w)

        # w_q = w_q.reshape(org_w_shape)
        linear_layer._weight_quantizer.observer.reset_min_max_vals()
        linear_layer._weight_quantizer.observer.to(self.device)
        for module in linear_layer.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
                module.disable_fake_quant()

        if get_scale_zp:
            linear_layer.weight.data = w_q
        return cast(torch.Tensor, w_q)

    def init_quant(self) -> Tuple[nn.ModuleList, Dict[str, Any], torch.Tensor]:
        modules = self.awq_model.get_model_layers(self.model, self.model_decoder_layers)
        batch_inps = []
        for sample in self.data_loader:
            if isinstance(sample, Dict):
                batch_inps.append(sample['input_ids'])
            elif isinstance(sample, torch.Tensor):
                batch_inps.append(sample)

        batch_inps = torch.cat(batch_inps, dim=0)
        batched_samples = {}
        batched_samples['input_ids'] = batch_inps
        data_loader_iter = iter(self.data_loader)
        first_batch = next(data_loader_iter)
        if not isinstance(first_batch, torch.Tensor):
            for k, v in first_batch.items():
                if k == 'input_ids':
                    continue
                batched_samples[k] = v
        modules, layer_kwargs, inputs = cache_model_inps(self.awq_model, self.model, modules, [batched_samples],
                                                         self.embedding_layers, self.device)
        return modules, layer_kwargs, inputs[0]

    def _apply_quant(self, named_linears: Dict[str, nn.Linear]) -> None:
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(self.device)
            self.pseudo_quantize_tensor(linear_layer.weight.data, linear_layer, get_scale_zp=True)
