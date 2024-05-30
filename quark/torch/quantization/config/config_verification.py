#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any
from dataclasses import fields

from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype
from quark.torch.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


def verify_quantization_spec(any_config: Any) -> None:
    if isinstance(any_config, QuantizationSpec):
        check_quantization_spec(any_config)
    elif hasattr(any_config, '__dataclass_fields__'):
        for field in fields(any_config):
            field_value = getattr(any_config, field.name)
            verify_quantization_spec(field_value)
    elif isinstance(any_config, dict):
        for value in any_config.values():
            verify_quantization_spec(value)
    elif isinstance(any_config, (list, tuple)):
        for item in any_config:
            verify_quantization_spec(item)


def check_quantization_spec(quantization_spec: QuantizationSpec) -> None:
    if quantization_spec.dtype in [Dtype.int8, Dtype.int4, Dtype.uint4, Dtype.fp8_e4m3]:
        if quantization_spec.is_dynamic is None:
            logger.error(
                f"The is_dynamic cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
        if quantization_spec.observer_cls is None:
            logger.error(
                f"The observer_cls cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
        if quantization_spec.qscheme is None:
            logger.error(
                f"The qscheme cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
    if quantization_spec.dtype in [Dtype.int8, Dtype.int4, Dtype.uint4]:
        if quantization_spec.symmetric is None:
            logger.error(
                f"The symmetric cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
        if quantization_spec.round_method is None:
            logger.error(
                f"The round_method cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
        if quantization_spec.scale_type is None:
            logger.error(
                f"The scale_type cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly.",
                error_code="Quant-Config")
    if quantization_spec.dtype in [Dtype.float16, Dtype.bfloat16]:
        for field in fields(quantization_spec):
            quant_config_spec = getattr(quantization_spec, field.name)
            if (field.name not in ['observer_cls', 'dtype']) and \
                    (quant_config_spec is not None):
                logger.warning(
                    f"{field.name} will not be used when Dtype is {quantization_spec.dtype.name} in QuantizationSpec.")


def init_quantization_config(quantization_config: QuantizationConfig) -> tuple[bool, bool]:
    is_dynamic = True
    is_weight_only = True
    for field in fields(QuantizationConfig):
        quantization_spec = getattr(quantization_config, field.name)
        if isinstance(quantization_spec, QuantizationSpec):
            if quantization_spec.dtype in [Dtype.int8, Dtype.int4, Dtype.uint4, Dtype.fp8_e4m3
                                           ] and quantization_spec.is_dynamic is False:
                is_dynamic = False
            if field.name in ["input_tensors", "output_tensors"
                              ] and quantization_spec.dtype not in [Dtype.float16, Dtype.bfloat16]:
                is_weight_only = False
    return is_dynamic, is_weight_only
