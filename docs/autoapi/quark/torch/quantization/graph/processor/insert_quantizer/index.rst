:orphan:

:py:mod:`quark.torch.quantization.graph.processor.insert_quantizer`
===================================================================

.. py:module:: quark.torch.quantization.graph.processor.insert_quantizer


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.processor.insert_quantizer.insert_quantizer



.. py:function:: insert_quantizer(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   Inserts FakeQuantize `call_module` nodes in the graph for input and/or output quantization, if necessary, based on the `quantization_annotation` metadata attached to nodes.


