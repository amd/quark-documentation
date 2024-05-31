#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Union, List, Dict
import torch
from torch.utils.data import DataLoader
if TYPE_CHECKING:
    from quark.torch.algorithm.models.base import BaseQuantModelForCausalLM
from transformers import PreTrainedModel


class BaseAlgoProcessor(ABC):

    @abstractmethod
    def __init__(
        self, q_model: BaseQuantModelForCausalLM, model: PreTrainedModel, quant_algo_config: Any,
        calib_data: Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]],
                          DataLoader[Dict[str, torch.Tensor]]]
    ) -> None:
        pass

    @abstractmethod
    def apply(self) -> None:
        pass
