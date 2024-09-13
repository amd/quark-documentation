:py:mod:`quark.torch.extensions.brevitas.config`
================================================

.. py:module:: quark.torch.extensions.brevitas.config

.. autoapi-nested-parse::

   Quark Quantization Config API for Brevitas.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.extensions.brevitas.config.Backend
   quark.torch.extensions.brevitas.config.Config
   quark.torch.extensions.brevitas.config.QuantizationConfig
   quark.torch.extensions.brevitas.config.QuantType
   quark.torch.extensions.brevitas.config.ParamType
   quark.torch.extensions.brevitas.config.QuantizationSpec




.. py:class:: Backend




   The backend target for quantization:
   - `layerwise`: Only quantizes inputs and weights of compute-heavy layers.


.. py:class:: Config


   A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

   - `global_quant_config`: The quantization configuration to be applied to the entire model.
   - `pre_quant_opt_config`: Optional optimization and pre-processing algorithms to apply to the model before quantization.
   - `algo_config`: optional algorithms to apply to the model after quantization to improve accuracy.
   - `backend`: The quantization backend to use.


.. py:class:: QuantizationConfig


   A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

   - `input_tensors`: The quantization parameters (if any) to apply to activation inputs.
   - `output_tensors`: The quantization parameters (if any) to apply to activation outputs.
   - `weight`: The quantization parameters (if any) to apply to the model weights.
   - `bias`: The quantization parameters (if any) to apply to the model biases.


.. py:class:: QuantType




   The fundamental data type of the quantized values:

   - `int_quant`: Values quantized to integers.
   - `float_quant`: Values quantized to floating point.



.. py:class:: ParamType




   Method for determining scale and zero point:

   - `stats`: Statistics
   - `mse`: Mean Squared Error


.. py:class:: QuantizationSpec


   A data class that defines the specifications for quantizing tensors within a model.
   It has some reasonable defaults so it can be used as is if desired.

   - `qscheme`: The granularity of quantization e.g. if applied to the whole tensor or to each channel.
   - `symmetric`: If true, the zero point is in the middle of the range of representable numbers, if false the quantized value will be mapped to between the minimum and maximum observed values. Asymmetric quantization is more expensive but may be better for ranges that aren't expected to be negative.
   - `scale_type`: Whether the scales use floating point or power of two values. Power of two allows lower bit widths and may be required by some embedded devices.
   - `quant_type`: The type of quantization we want: integer or floating point. If float, we also need to specify the exponent and mantissa bit widths.
   - `param_type`: Method for determing scale and zero point.
   - `bit_width`: Level of precision we want the quantization to be.
   - `exponent_bit_width`: The level of precision we want for the exponent when using the float quant_type.
   - `mantissa_bit_width`: The level of precision we want for the mantissa when using the float quant_type.


