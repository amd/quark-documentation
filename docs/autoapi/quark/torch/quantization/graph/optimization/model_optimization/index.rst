:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.model_optimization`
========================================================================

.. py:module:: quark.torch.quantization.graph.optimization.model_optimization


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.model_optimization.trans_opsfunc_2_quant_module
   quark.torch.quantization.graph.optimization.model_optimization.apply_pre_hw_constrain_passes
   quark.torch.quantization.graph.optimization.model_optimization.apply_post_hw_constrain_passes



.. py:function:: trans_opsfunc_2_quant_module(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   optimize the pure torch.ops.aten.*** functional model,
    replace the




