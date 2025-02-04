:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.pre_quant.replace_convtranspose2d_to_qtconvtranspose2d`
============================================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.pre_quant.replace_convtranspose2d_to_qtconvtranspose2d


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.pre_quant.replace_convtranspose2d_to_qtconvtranspose2d.replace_convtranspose2d_qtconvtranspose2d



.. py:function:: replace_convtranspose2d_qtconvtranspose2d(m: torch.fx.GraphModule) -> torch.fx.GraphModule

   replace [ops.aten.conv_transpose2d.input] to QuantConvTranspose2d
   ops.conv_transpose2d.input:
       args: (Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1) -> Tensor
       required: [input, weight]
       optional: [bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] output_padding=0, SymInt groups=1, SymInt[2] dilation=1]


