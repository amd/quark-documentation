:orphan:

:py:mod:`quark.torch.quantization.nn.modules.quantize_conv_bn_fused`
====================================================================

.. py:module:: quark.torch.quantization.nn.modules.quantize_conv_bn_fused


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.nn.modules.quantize_conv_bn_fused.QuantizedConvBatchNorm2d



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.nn.modules.quantize_conv_bn_fused.update_bn_stats
   quark.torch.quantization.nn.modules.quantize_conv_bn_fused.freeze_bn_stats
   quark.torch.quantization.nn.modules.quantize_conv_bn_fused.fuse_conv_bn
   quark.torch.quantization.nn.modules.quantize_conv_bn_fused.clear_non_native_bias



.. py:class:: QuantizedConvBatchNorm2d(in_channels: int, out_channels: int, kernel_size: torch.nn.common_types._size_2_t, stride: torch.nn.common_types._size_2_t = 1, padding: torch.nn.common_types._size_2_t = 0, dilation: torch.nn.common_types._size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', eps: float = 1e-05, momentum: float = 0.1, freeze_bn_stats: bool = False, quant_config: Optional[quark.torch.quantization.config.config.QuantizationConfig] = QuantizationConfig())




   A QuantizedConvBatchNorm2d module is a module fused from
       Conv2d and BatchNorm2d attached with FakeQuantizer modules for weight and
       batchnorm stuffs used in quantization aware training.

       We combined the interface of :class:`torch.nn.Conv2d` and
       :class:`torch.nn.BatchNorm2d`.

       Implementation details: https://arxiv.org/pdf/1806.08342.pdf section 3.2.2

       Similar to :class:`torch.nn.Conv2d`, with FakeQuantizer modules initialized
       to default.
   #     






