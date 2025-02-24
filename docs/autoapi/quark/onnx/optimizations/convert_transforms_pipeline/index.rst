:py:mod:`quark.onnx.optimizations.convert_transforms_pipeline`
==============================================================

.. py:module:: quark.onnx.optimizations.convert_transforms_pipeline

.. autoapi-nested-parse::

   Transformations pipeline for onnx model conversion.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.optimizations.convert_transforms_pipeline.ConvertQDQToQOPTransformsPipeline
   quark.onnx.optimizations.convert_transforms_pipeline.RemoveQDQTransformsPipeline




.. py:class:: ConvertQDQToQOPTransformsPipeline(configs: Optional[Dict[str, Any]] = None)




   Convert QDQ to QOperator transformations pipeline.

   .. py:method:: apply(model: onnx.ModelProto, candidate_nodes: Any, node_metadata: Any) -> Tuple[onnx.ModelProto, Any]

      Implement the transforms.

      Args:
          model: Onnx model to be quantized.

      Returns:
          Conveted onnx model.



.. py:class:: RemoveQDQTransformsPipeline(configs: Optional[Dict[str, Any]] = None)




   Remove QDQ pairs transformations pipeline.

   .. py:method:: apply(model: onnx.ModelProto, candidate_nodes: Any, node_metadata: Any) -> Tuple[onnx.ModelProto, Any]

      Implement the transforms.

      Args:
          model: Onnx model to be quantized.

      Returns:
          Conveted onnx model.



