:orphan:

:py:mod:`quark.torch.quantization.graph.processor.processor`
============================================================

.. py:module:: quark.torch.quantization.graph.processor.processor


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.processor.processor.transform_for_annotation
   quark.torch.quantization.graph.processor.processor.freeze_model
   quark.torch.quantization.graph.processor.processor.mark_exclude_nodes



.. py:function:: transform_for_annotation(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   Prepare before annotation, for both PTQ and QAT


.. py:function:: freeze_model(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   After quantization, we need to export model (e.g onnx, torch.export),
   we regard the users will not need further calibration, training, optimization.


.. py:function:: mark_exclude_nodes(model: torch.fx.GraphModule, exclude: List[str]) -> List[str]

   Attaches `skip_quant` metadata to FX nodes to specify which nodes should not be quantized based on the list
   of patterns `exclude`.


