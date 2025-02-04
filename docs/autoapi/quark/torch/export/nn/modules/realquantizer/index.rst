:orphan:

:py:mod:`quark.torch.export.nn.modules.realquantizer`
=====================================================

.. py:module:: quark.torch.export.nn.modules.realquantizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.nn.modules.realquantizer.RealQuantizerBase
   quark.torch.export.nn.modules.realquantizer.ScaledRealQuantizer
   quark.torch.export.nn.modules.realquantizer.NonScaledRealQuantizer




.. py:class:: RealQuantizerBase




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: ScaledRealQuantizer(qspec: quark.torch.quantization.config.config.QuantizationSpec, quantizer: Optional[quark.torch.quantization.tensor_quantize.FakeQuantizeBase], reorder: bool, real_quantized: bool, float_dtype: torch.dtype, device: Optional[torch.device] = torch.device('cuda'), scale_shape: Optional[Tuple[int, Ellipsis]] = None, zero_point_shape: Optional[Tuple[int, Ellipsis]] = None)




   On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
   On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.

   .. py:method:: to_real_quantize_params(param: torch.Tensor) -> torch.Tensor

      Quantize weight and bias on low-bit precision datatypes, and pack them if required.



.. py:class:: NonScaledRealQuantizer(qspec: quark.torch.quantization.config.config.QuantizationSpec, quantizer: Optional[quark.torch.quantization.tensor_quantize.FakeQuantizeBase], reorder: bool, real_quantized: bool, float_dtype: torch.dtype, device: Optional[torch.device] = torch.device('cuda'), scale_shape: Optional[Tuple[int, Ellipsis]] = None, zero_point_shape: Optional[Tuple[int, Ellipsis]] = None)




   On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
   On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.

   .. py:method:: to_real_quantize_params(param: torch.Tensor) -> torch.Tensor

      Quantize weight and bias on low-bit precision datatypes, and pack them if required.



