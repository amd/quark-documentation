#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig, AWQConfig, SmoothQuantConfig, GPTQConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerChannelMinMaxObserver

# Bfloat16 config
BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PerTensorMinMaxObserver)

# Float16 config
FLOAT16_SPEC = QuantizationSpec(dtype=Dtype.float16, observer_cls=PerTensorMinMaxObserver)
DEFAULT_FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

# Fp8(e4m3) config
FP8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                       qscheme=QSchemeType.per_tensor,
                                       observer_cls=PerTensorMinMaxObserver,
                                       is_dynamic=False)
DEFAULT_W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                           weight=FP8_PER_TENSOR_SPEC)

DEFAULT_W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC,
                                                                weight=FP8_PER_TENSOR_SPEC,
                                                                output_tensors=FP8_PER_TENSOR_SPEC)

# Per tensor config
INT4_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)

INT8_PER_TENSER_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)

INT8_PER_TENSER_DYNAMIC_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                                qscheme=QSchemeType.per_tensor,
                                                observer_cls=PerTensorMinMaxObserver,
                                                symmetric=True,
                                                scale_type=ScaleType.float,
                                                round_method=RoundType.half_even,
                                                is_dynamic=True)

DEFAULT_W_INT4_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT4_PER_TENSER_SPEC)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_SPEC,
                                                             weight=INT8_PER_TENSER_SPEC)

DEFAULT_W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSER_DYNAMIC_SPEC,
                                                                     weight=INT8_PER_TENSER_DYNAMIC_SPEC)

# Per Channel Config
INT4_PER_CHANNEL_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                         observer_cls=PerChannelMinMaxObserver,
                                         symmetric=True,
                                         scale_type=ScaleType.float,
                                         round_method=RoundType.half_even,
                                         qscheme=QSchemeType.per_channel,
                                         ch_axis=0,
                                         is_dynamic=False)

DEFAULT_W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)

# Per Group Config
DEFAULT_INT4_PER_GROUP_SYM_SPEC = QuantizationSpec(dtype=Dtype.int4,
                                                   observer_cls=PerChannelMinMaxObserver,
                                                   symmetric=True,
                                                   scale_type=ScaleType.float,
                                                   round_method=RoundType.half_even,
                                                   qscheme=QSchemeType.per_group,
                                                   ch_axis=0,
                                                   is_dynamic=False,
                                                   group_size=128)

DEFAULT_W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=DEFAULT_INT4_PER_GROUP_SYM_SPEC)

DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                     observer_cls=PerChannelMinMaxObserver,
                                                     symmetric=False,
                                                     scale_type=ScaleType.float,
                                                     round_method=RoundType.half_even,
                                                     qscheme=QSchemeType.per_group,
                                                     ch_axis=0,
                                                     is_dynamic=False,
                                                     group_size=128)

DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
DEFAULT_W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QuantizationConfig(input_tensors=BFLOAT16_SPEC,
                                                                 weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)

# Default AWQ Config
DEFAULT_AWQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=AWQConfig())

# Default SmoothQuant Config
DEFAULT_SMOOTH_QUANT_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG,
                                     algo_config=SmoothQuantConfig())

# Default GPTQ Config
DEFAULT_GPTQ_CONFIG = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG, algo_config=GPTQConfig())
