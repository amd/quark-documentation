:orphan:

:py:mod:`quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant`
=====================================================================================

.. py:module:: quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant.SplitQuantModuleCalledOverOnce
   quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant.ConvertBn2D2ConvQOPass
   quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant.ConvertReduceMean2GapQOPass




.. py:class:: SplitQuantModuleCalledOverOnce




   Base interface for implementing passes.

   It is required to implement the `call` function so that we can directly
   pass instances of the Pass directly to the PassManager and call them as a
   function.

   We can directly pass an instance of a class implementing this interface into
   the PassManager's `passes` attribute.

   .. py:method:: requires(graph_module: torch.fx.GraphModule) -> None

      This function will be called before the pass is run and will check that
      the given graph module contains the preconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on


   .. py:method:: call(m: torch.fx.GraphModule) -> torch.fx.GraphModule

      For better deployment for AMD's specific hardware, e.g IPU
      if one module used over one in forward, we will instance a quant module for each all proceduce



.. py:class:: ConvertBn2D2ConvQOPass




   Base interface for implementing passes.

   It is required to implement the `call` function so that we can directly
   pass instances of the Pass directly to the PassManager and call them as a
   function.

   We can directly pass an instance of a class implementing this interface into
   the PassManager's `passes` attribute.

   .. py:method:: requires(graph_module: torch.fx.GraphModule) -> None

      This function will be called before the pass is run and will check that
      the given graph module contains the preconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on


   .. py:method:: call(m: torch.fx.GraphModule) -> torch.fx.GraphModule

      process a single bn layer (with no conv2d before)
      transfer the bn layer to a single conv2d node



.. py:class:: ConvertReduceMean2GapQOPass




   For torch code: is torch.mean( **args) is equal to torch.nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling
   for the corresponding ONNX model: change reduce_mean type node to GlobalAveragePooling type node
   change reduce mean to global average pooling if they are equal.
    NOTE at present support 2D image/feature  [N, C,H, W]

   .. py:method:: requires(graph_module: torch.fx.GraphModule) -> None

      This function will be called before the pass is run and will check that
      the given graph module contains the preconditions needed to run the
      pass. It is not required to implement this function.

      Args:
          graph_module: The graph module we will run checks on


   .. py:method:: call(m: torch.fx.GraphModule) -> torch.fx.GraphModule

      if a torch.ops.aten.mean.dim() equal to torch.ops.aten.adaptive_avg_pool2d.default(x, [1, 1])
      then change, to align with ONNX strategy, to let the final onnx model to GlobalAveragePooling node



