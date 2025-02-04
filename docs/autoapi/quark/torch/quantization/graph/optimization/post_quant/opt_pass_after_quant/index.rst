:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant`
=====================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant.ConvertClip2ReLUQOPass




.. py:class:: ConvertClip2ReLUQOPass




   This is a post-quantization optimization.
   after quantization, we get a model as follows:
       ...
       x = torch.clip(x, clip_min=num1, clip_max=num2)
       x = fake_quantizer(x)
       ...
       x = torch.clip(x, clip_min=num3, clip_max=num4)
       x = other_type_layer(x) # not fake_quantizer
   post quant optimization:
       the clip that satisfies some condition can be transferred to ReLU layer:
           1. following a fake quantizer layer
           2. clip_min =  0

   .. py:method:: requires(graph_module: torch.fx.GraphModule) -> None

      This function will be called before the pass is run and will check that
      the given graph module contains the preconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on


   .. py:method:: call(m: torch.fx.GraphModule) -> torch.fx.GraphModule

      convert a clip layer to relu layer
      (only activate under condition that relu can be act as clip)



