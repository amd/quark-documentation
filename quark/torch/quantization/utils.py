#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional
import gc
import torch
from quark.torch.quantization.config.type import Dtype


def clear_memory(weight: Optional[torch.Tensor] = None) -> None:
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    assert (quant_min < quant_max), "qmin must be less than qmax."


def calculate_qmin_qmax(dtype: Dtype) -> tuple[Optional[int], Optional[int]]:
    # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
    if dtype == Dtype.int8:
        quant_min, quant_max = -128, 127
    elif dtype == Dtype.int4:
        quant_min, quant_max = -8, 7
    elif dtype == Dtype.uint4:
        quant_min, quant_max = 0, 15
    else:
        quant_min, quant_max = None, None
    return quant_min, quant_max
