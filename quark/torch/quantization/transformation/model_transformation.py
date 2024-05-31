#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
from typing import Union
import torch.nn as nn


def set_op_by_name(layer: Union[nn.Module, nn.ModuleList], name: str, new_module: nn.Module) -> None:
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_op_by_name(layer: Union[nn.Module, nn.ModuleList], name: str) -> Union[nn.Module, nn.ModuleList]:
    levels = name.split('.')
    mod_ = layer
    for l_idx in range(len(levels)):
        if levels[l_idx].isdigit() and isinstance(mod_, nn.ModuleList):
            mod_ = mod_[int(levels[l_idx])]
        else:
            mod_ = getattr(mod_, levels[l_idx])
    return mod_
