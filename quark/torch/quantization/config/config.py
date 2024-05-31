#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for PyTorch"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, Optional, Dict, List, Type
from quark.torch.quantization.observer.observer import ObserverBase
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType, DeviceType
from quark.torch.algorithm.awq.awq import AwqProcessor


@dataclass(eq=True)
class Config:
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
    :param Dict[str, QuantizationConfig] layer_type_quant_config: A dictionary mapping from layer types (e.g., 'Conv2D', 'Dense') to their quantization configurations. Default is an empty dictionary.
    :param Dict[str, QuantizationConfig] layer_quant_config: A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization. Default is an empty dictionary.
    :param List[str] exclude: A list of layer names to be excluded from quantization, enabling selective quantization of the model. Default is an empty list.
    :param Optional[AlgoConfig] algo_config: Optional configuration for the quantization algorithm, such as GPTQ and AWQ. After this process, the datatype/fake_datatype of weights will be changed with quantization scales. Default is None.
    :param Optional[Union[PreQuantOptConfig, List[PreQuantOptConfig]]] pre_quant_opt_config: Optional pre-processing optimization, such as Equalization and SmoothQuant. After this process, the value of weights will be changed, but the dtype/fake_dtype will be the same. Default is None.
    """

    # Global quantization configuration applied to the entire model unless overridden at the layer level.
    global_quant_config: QuantizationConfig

    # A dictionary mapping from layer types (e.g., 'Conv2D', 'Dense') to their quantization configurations.
    layer_type_quant_config: Dict[str, QuantizationConfig] = field(default_factory=dict)

    # A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization.
    layer_quant_config: Dict[str, QuantizationConfig] = field(default_factory=dict)

    # A list of layer names to be excluded from quantization, enabling selective quantization of the model.
    exclude: List[str] = field(default_factory=list)

    # Optional configuration for the quantization algorithm, such as GPTQ and AWQ
    # After this process, the datatype/fake_datatype of weights will be changed with quantization scales.
    algo_config: Optional[AlgoConfig] = None

    # Optional pre-processing optimization, such as Equalization and SmoothQuant.
    # After this process, the value of weights will be changed, but the dtype/fake_dtype will be the same.
    pre_quant_opt_config: Optional[Union[PreQuantOptConfig, List[PreQuantOptConfig]]] = None


@dataclass(eq=True)
class QuantizationConfig:
    """
    A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

    :param Optional[QuantizationSpec] input_tensors: Input tensors quantization specification. If None, following the hierarchical quantization setup. e.g. If the input_tensors in layer_type_quant_config is None, the configuration from global_quant_config will be used instead. Defaults to None. If None in global_quant_config, input_tensors are not quantized.
    :param Optional[QuantizationSpec] output_tensors: Output tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[QuantizationSpec] weight: The weights tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[QuantizationSpec] bias: The bias tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[DeviceType] target_device: Configuration specifying the target device (e.g., CPU, GPU, IPU) for the quantized model.

    """

    input_tensors: Optional[QuantizationSpec] = None

    output_tensors: Optional[QuantizationSpec] = None

    weight: Optional[QuantizationSpec] = None

    bias: Optional[QuantizationSpec] = None

    target_device: Optional[DeviceType] = None


@dataclass(eq=True, frozen=True)
class QuantizationSpec:
    """
    A data class that defines the specifications for quantizing tensors within a model.

    :param Dtype dtype: The data type for quantization (e.g., int8, int4).
    :param Optional[bool] is_dynamic: Specifies whether dynamic or static quantization should be used. Default is None, which indicates no specification.
    :param Optional[Type[ObserverBase]] observer_cls: The class of observer to be used for determining quantization parameters like min/max values. Default is None.
    :param Optional[QSchemeType] qscheme: The quantization scheme to use, such as per_tensor, per_channel or per_group. Default is None.
    :param Optional[int] ch_axis: The channel axis for per-channel quantization. Default is None.
    :param Optional[int] group_size: The size of the group for per-group quantization. Default is None.
    :param Optional[bool] symmetric: Indicates if the quantization should be symmetric around zero. If True, quantization is symmetric. If None, it defers to a higher-level or global setting. Default is None.
    :param Optional[RoundType] round_method: The rounding method during quantization, such as half_even. If None, it defers to a higher-level or default method. Default is None.
    :param Optional[ScaleType] scale_type: Defines the scale type to be used for quantization, like power of two or float. If None, it defers to a higher-level setting or uses a default method. Default is None.

    """

    ###################################################################################################
    # Quantization Specification for Dtype in [Bfloat16, FP8, Int]

    dtype: Dtype

    ###################################################################################################
    # Quantization Specification for Dtype in [FP8, Int]

    is_dynamic: Optional[bool] = None

    observer_cls: Optional[Type[ObserverBase]] = None

    qscheme: Optional[QSchemeType] = None

    ch_axis: Optional[int] = None

    group_size: Optional[int] = None

    ###################################################################################################
    # Quantization Specification for Dtype in [Int]

    symmetric: Optional[bool] = None

    round_method: Optional[RoundType] = None

    scale_type: Optional[ScaleType] = None

    def __post_init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[str, int, bool, None]]) -> QuantizationSpec:
        dtype = Dtype[config_dict["dtype"]]  # type:ignore

        if config_dict.get("qscheme", None) is not None:
            qscheme = QSchemeType[config_dict["qscheme"]]  # type:ignore
        else:
            qscheme = None

        if config_dict.get("round_method", None) is not None:
            round_method = RoundType[config_dict["round_method"]]  # type:ignore
        else:
            round_method = None

        if config_dict.get("scale_type", None) is not None:
            scale_type = ScaleType[config_dict["scale_type"]]  # type:ignore
        else:
            scale_type = None

        is_dynamic = config_dict["is_dynamic"] if config_dict.get("is_dynamic", None) is not None else None
        ch_axis = config_dict["ch_axis"] if config_dict.get("ch_axis", None) is not None else None
        group_size = config_dict["group_size"] if config_dict.get("group_size", None) is not None else None
        symmetric = config_dict["symmetric"] if config_dict.get("symmetric", None) is not None else None
        return cls(
            dtype=dtype,
            is_dynamic=is_dynamic,  # type:ignore
            qscheme=qscheme,
            ch_axis=ch_axis,  # type:ignore
            group_size=group_size,  # type:ignore
            symmetric=symmetric,  # type:ignore
            round_method=round_method,
            scale_type=scale_type)


@dataclass
class SmoothQuantConfig:
    """
    A data class that defines the specifications for Smooth Quantization.

    :param str name: The name of the configuration, typically used to identify different quantization settings. Default is "smoothquant".
    :param int alpha: The factor of adjustment in the quantization formula, influencing how aggressively weights are quantized. Default is 1.
    :param float scale_clamp_min: The minimum scaling factor to be used during quantization, preventing the scale from becoming too small. Default is 1e-3.
    :param Optional[List[Dict[str, str]]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is None.
    :param Optional[List[str]] embedding_layers: A list of embedding layer names that require special quantization handling to maintain their performance and accuracy. Default is None.
    :param Optional[str] model_decoder_layers: Specifies any particular decoder layers in the model that might have unique quantization requirements. Default is None.
    """
    name: str = "smoothquant"
    alpha: int = 1
    scale_clamp_min: float = 1e-3
    scaling_layers: Optional[List[Dict[str, str]]] = None
    embedding_layers: Optional[List[str]] = None
    model_decoder_layers: Optional[str] = None


@dataclass
class EqualizationConfig:
    name: str = "equalization"


PreQuantOptConfig = Union[SmoothQuantConfig, EqualizationConfig]


@dataclass
class AWQConfig:
    """
    Configuration for Activation-aware Weight Quantization (AWQ).

    :param str name: The name of the quantization configuration. Default is "awq".
    :param int bit: The bit width for weights, indicating the precision of quantization. Defaults to 4 bits.
    :param bool sym: Indicates whether symmetric quantization should be used. If True, quantization is symmetric around zero. Default is False.
    :param int group_size: The size of the group for grouped quantization, specifying how many weights are quantized together using the same scale and zero-point. Default is 128.
    :param Type[AwqProcessor] algo_processor: The processor type that handles the AWQ algorithm logic.
    :param Optional[List[Dict[str, str]]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is None.
    :param Optional[str] model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is None.
    :param Optional[List[str]] embedding_layers: Lists the embedding layers within the model that need to be quantized separately. Default is None.
    """
    name: str = "awq"
    bit: int = 4  # The bit width for weights, default is 4 bits.
    sym: bool = False  # If symmetric quantization, default is False.
    group_size: int = 128  # The size of the group for grouped quantization, default is 128.
    algo_processor: Type[AwqProcessor] = AwqProcessor
    scaling_layers: Optional[List[Dict[str, str]]] = None
    model_decoder_layers: Optional[str] = None
    embedding_layers: Optional[List[str]] = None


@dataclass
class GPTQConfig:
    """
    A data class that defines the specifications for Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ).

    :param str name: The configuration name. Default is "gptq".
    :param int bit: The bit width for quantization, indicating the precision level of the quantized values. Defaults to 4 bits.
    :param bool sym: Specifies whether symmetric quantization is used. Symmetric quantization centers the quantized values around zero. Default is False.
    :param int group_size: Specifies the number of weights to be quantized together as a group, using the same scale and zero-point. Default is 128.
    :param float damp_percent: The percentage used to dampen the quantization effect, aiding in the maintenance of accuracy post-quantization. Default is 0.01.
    :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is True.
    :param bool static_groups: Specifies whether the groups for quantization are static (fixed during initialization) or can be dynamically adjusted during training. Default is False.
    :param bool true_sequential: Indicates whether the quantization should be applied in a truly sequential manner across the layers. Default is True.
    :param Optional[List[str]] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling. Default is None.
    :param Optional[str] model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model. Default is None.
    :param Optional[List[str]] embedding_layers: Identifies which embedding layers within the model need to be quantized separately to preserve embedding quality. Default is None.
    """
    name: str = "gptq"
    bit: int = 4
    sym: bool = False
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    static_groups: bool = False
    true_sequential: bool = True
    inside_layer_modules: Optional[List[str]] = None
    model_decoder_layers: Optional[str] = None
    embedding_layers: Optional[List[str]] = None


AlgoConfig = Union[SmoothQuantConfig, AWQConfig, GPTQConfig]
