:orphan:

:py:mod:`quark.torch.quantization.config.type`
==============================================

.. py:module:: quark.torch.quantization.config.type


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.config.type.QSchemeType
   quark.torch.quantization.config.type.Dtype
   quark.torch.quantization.config.type.ScaleType
   quark.torch.quantization.config.type.RoundType
   quark.torch.quantization.config.type.DeviceType
   quark.torch.quantization.config.type.QuantizationMode




.. py:class:: QSchemeType




   The quantization schemes applicable to tensors within a model.

   - `per_tensor`: Quantization is applied uniformly across the entire tensor.
   - `per_channel`: Quantization parameters differ across channels of the tensor.
   - `per_group`: Quantization parameters differ across defined groups of weight tensor elements.



.. py:class:: Dtype




   The data types used for quantization of tensors.

   - `int8`: Signed 8-bit integer, range from -128 to 127.
   - `uint8`: Unsigned 8-bit integer, range from 0 to 255.
   - `int4`: Signed 4-bit integer, range from -8 to 7.
   - `uint4`: Unsigned 4-bit integer, range from 0 to 15.
   - `bfloat16`: Bfloat16 format.
   - `float16`: Standard 16-bit floating point format.
   - `fp8_e4m3`: FP8 format with 4 exponent bits and 3 bits of mantissa.
   - `mx`: MX format 8 bit shared scale value with fp8 element data types.



.. py:class:: ScaleType




   The types of scales used in quantization.

   - `float`: Scale values are floating-point numbers.
   - `pof2`: Scale values are powers of two.



.. py:class:: RoundType




   The rounding methods used during quantization.

   - `round`: Rounds.
   - `floor`: Floors towards the nearest even number.
   - `half_even`: Rounds towards the nearest even number.



.. py:class:: DeviceType




   The target devices for model deployment and optimization.

   - `CPU`: CPU.
   - `IPU`: IPU.


.. py:class:: QuantizationMode




   Different quantization modes.

   - `eager_mode`: The eager mode based on PyTorch in-place operator replacement.
   - `fx_graph_mode`: The graph mode based on torch.fx.


