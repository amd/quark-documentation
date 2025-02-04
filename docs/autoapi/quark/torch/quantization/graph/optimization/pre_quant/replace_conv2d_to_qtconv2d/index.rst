:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.pre_quant.replace_conv2d_to_qtconv2d`
==========================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.pre_quant.replace_conv2d_to_qtconv2d


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.pre_quant.replace_conv2d_to_qtconv2d.replace_conv2d_qtconv2d



.. py:function:: replace_conv2d_qtconv2d(m: torch.fx.GraphModule) -> torch.fx.GraphModule

   replace [ops.aten.conv2d] to QuantConv2d
   ops.aten.conv2d:
       args: (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor
       required: [input, weight]
       optional: [bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1]


