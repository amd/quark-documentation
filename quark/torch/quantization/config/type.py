#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from enum import Enum, auto


class QSchemeType(Enum):
    """
    The quantization schemes applicable to tensors within a model.

    - `per_tensor`: Quantization is applied uniformly across the entire tensor.
    - `per_channel`: Quantization parameters differ across channels of the tensor.
    - `per_group`: Quantization parameters differ across defined groups of weight tensor elements.

    """

    per_tensor = auto(),
    per_channel = auto(),
    per_group = auto()


class Dtype(Enum):
    """
    The data types used for quantization of tensors.

    - `int8`: Signed 8-bit integer, range from -128 to 127.
    - `int4`: Signed 4-bit integer, range from -8 to 7.
    - `uint4`: Unsigned 4-bit integer, range from 0 to 15.
    - `fp8_e4m3`: FP8 format with 4 exponent bits and 3 bits of mantissa.
    - `bfloat16`: Bfloat16 format.
    - `float16`: Standard 16-bit floating point format.

    """
    int8 = auto(),
    int4 = auto(),
    uint4 = auto(),
    fp8_e4m3 = auto(),
    bfloat16 = auto(),
    float16 = auto()


class ScaleType(Enum):
    """
    The types of scales used in quantization.

    - `float`: Scale values are floating-point numbers.
    - `pof2`: Scale values are powers of two.

    """
    float = auto(),
    pof2 = auto()


class RoundType(Enum):
    """
    The rounding methods used during quantization.

    - `half_even`: Rounds towards the nearest even number.

    """
    half_even = 8


class DeviceType(Enum):
    """
    The target devices for model deployment and optimization.

    - `CPU`: CPU.
    - `IPU`: IPU.
    """
    CPU = "cpu"
    IPU = "ipu"
