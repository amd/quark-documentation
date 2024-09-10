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



.. py:function:: transform_for_annotation(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   Prepare before annotation, for both PTQ and QAT


.. py:function:: freeze_model(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   After quantization, we need to export model (e.g onnx, torch.export),
   we regard the users will not need further calibration, training, optimization.


