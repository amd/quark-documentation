:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear`
==========================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear.replace_linear_qtlinear



.. py:function:: replace_linear_qtlinear(m: torch.fx.GraphModule) -> torch.fx.GraphModule

   replace [ops.aten.linear] to QuantLinear
   ops.aten.linear:
       args: (Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
       required: [input, weight]
       optional: [bias=None]


