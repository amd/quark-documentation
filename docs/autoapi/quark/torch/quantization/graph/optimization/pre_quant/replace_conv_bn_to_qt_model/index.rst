:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model`
===========================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model.replace_conv2dbn_quantizedconv_module



.. py:function:: replace_conv2dbn_quantizedconv_module(m: torch.fx.GraphModule) -> torch.fx.GraphModule

   replace [ops.aten.conv2d -> ops.aten.cudnn_batch_norm] to QuantizedConvBatchNorm2d(QAT)
   ops.aten.conv2d:
       args: (Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1)
       required: [input, weight]
       optional: [bias=None, stride=[1,1], padding=[0,0], dilation=[1,1], groups=1]
   cudnn_batch_norm:
       args: (Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
       required: [input, weight]
       optional: [bias, running_mean, running_var, training]


