#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional
import gc
import torch


def clear_memory(weight: Optional[torch.Tensor] = None) -> None:
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()
