:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.modify_reshape_param`
==========================================================================

.. py:module:: quark.torch.quantization.graph.optimization.modify_reshape_param


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.modify_reshape_param.modify_reshape_param



.. py:function:: modify_reshape_param(m: torch.fx.GraphModule) -> torch.fx.GraphModule

   In some case, reshape param in fx graph is traced by input datashape.
   For example: reshape.default(conv2d_85, [25, 80, 6400]) # where 25 is batch size
   we can change: [25, 80, 6400] to [-1, 80, 6400]


