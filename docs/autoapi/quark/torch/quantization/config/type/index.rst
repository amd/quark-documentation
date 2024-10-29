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
   quark.torch.quantization.config.type.ZeroPointType
   quark.torch.quantization.config.type.Dtype
   quark.torch.quantization.config.type.ScaleType
   quark.torch.quantization.config.type.RoundType
   quark.torch.quantization.config.type.DeviceType
   quark.torch.quantization.config.type.QuantizationMode
   quark.torch.quantization.config.type.TQTThresholdInitMeth




.. py:class:: QSchemeType(*args, **kwds)




   The quantization schemes applicable to tensors within a model.

   - `per_tensor`: Quantization is applied uniformly across the entire tensor.
   - `per_channel`: Quantization parameters differ across channels of the tensor.
   - `per_group`: Quantization parameters differ across defined groups of weight tensor elements.



.. py:class:: ZeroPointType(*args, **kwds)




   The zero point Dtype used for zero point.

   - 'int32': int zero point
   - 'float32': float zero point


.. py:class:: Dtype(*args, **kwds)




   The data types used for quantization of tensors.

   - `int8`: Signed 8-bit integer, range from -128 to 127.
   - `uint8`: Unsigned 8-bit integer, range from 0 to 255.
   - `int4`: Signed 4-bit integer, range from -8 to 7.
   - `uint4`: Unsigned 4-bit integer, range from 0 to 15.
   - `bfloat16`: Bfloat16 format.
   - `float16`: Standard 16-bit floating point format.
   - `fp8_e4m3`: FP8 format with 4 exponent bits and 3 bits of mantissa.
   - `fp8_e5m2`: FP8 format with 5 exponent bits and 2 bits of mantissa.
   - `fp6_e3m2`: FP6 format with 3 exponent bits and 2 bits of mantissa.
   - `fp6_e2m3`: FP6 format with 2 exponent bits and 3 bits of mantissa.
   - `fp4`: FP4 format.
   - `mx`: MX format 8 bit shared exponent with specific element data types.
   - `mx6`, `mx9`: Block data representation with multi-level ultra-fine scaling factors.



.. py:class:: ScaleType(*args, **kwds)




   The types of scales used in quantization.

   - `float`: Scale values are floating-point numbers.
   - `pof2`: Scale values are powers of two.



.. py:class:: RoundType(*args, **kwds)




   The rounding methods used during quantization.

   - `round`: Rounds.
   - `floor`: Floors towards the nearest even number.
   - `half_even`: Rounds towards the nearest even number.



.. py:class:: DeviceType(*args, **kwds)




   The target devices for model deployment and optimization.

   - `CPU`: CPU.
   - `IPU`: IPU.


.. py:class:: QuantizationMode(*args, **kwds)




   Different quantization modes.

   - `eager_mode`: The eager mode based on PyTorch in-place operator replacement.
   - `fx_graph_mode`: The graph mode based on torch.fx.


.. py:class:: TQTThresholdInitMeth(*args, **kwds)




   The method of threshold initialization of TQT algorithm in QAT. See Table 2 in https://arxiv.org/pdf/1903.08066.pdf

   - `_3SD`: The method of threshold initialization with std and 3 as hyperparameters.
   - `_LL_J`: The method of threshold initialization in the Algorithm 1 of paper "Quantizing Convolutional Neural Networks for Low-Power High-Throughput Inference Engines" - Sean Settle et al. https://arxiv.org/pdf/1805.07941.pdf


