:orphan:

:py:mod:`quark.torch.quantization.nn.modules.quantize_conv`
===========================================================

.. py:module:: quark.torch.quantization.nn.modules.quantize_conv


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.nn.modules.quantize_conv.QuantConv2d
   quark.torch.quantization.nn.modules.quantize_conv.QuantConvTranspose2d




.. py:class:: QuantConv2d(in_channels: int, out_channels: int, kernel_size: torch.nn.common_types._size_2_t, stride: torch.nn.common_types._size_2_t = 1, padding: torch.nn.common_types._size_2_t = 0, dilation: torch.nn.common_types._size_2_t = 1, output_padding: torch.nn.common_types._size_2_t = 0, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', quant_config: quark.torch.quantization.config.config.QuantizationConfig = QuantizationConfig(), reload: bool = False, device: torch.device = torch.device('cpu'))




   Quantized version of nn.Conv2d

       


.. py:class:: QuantConvTranspose2d(in_channels: int, out_channels: int, kernel_size: torch.nn.common_types._size_2_t, stride: torch.nn.common_types._size_2_t = 1, padding: torch.nn.common_types._size_2_t = 0, output_padding: torch.nn.common_types._size_2_t = 0, groups: int = 1, bias: bool = True, dilation: torch.nn.common_types._size_2_t = 1, padding_mode: str = 'zeros', quant_config: quark.torch.quantization.config.config.QuantizationConfig = QuantizationConfig(), reload: bool = False, device: torch.device = torch.device('cpu'))




   Quantized version of nn.ConvTranspose2d
       


