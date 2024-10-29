:py:mod:`quark.onnx.graph_transformations.transforms_pipeline`
==============================================================

.. py:module:: quark.onnx.graph_transformations.transforms_pipeline

.. autoapi-nested-parse::

   Abstract Base Class for model transformations pipeline to a onnx model.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.graph_transformations.transforms_pipeline.TransformsPipeline




.. py:class:: TransformsPipeline(configs: Optional[Dict[str, Any]] = None)




   Wrapper of transforms to the model, apply in sequence.
   Transforms the original model to perform better during quantization.

   .. py:method:: get_configs() -> Optional[Dict[str, Any]]

      Get the configurations.

      Args:
          None
      Returns:
          Dict of configurations


   .. py:method:: apply(model: onnx.ModelProto, candidate_layers: Any, layer_metadata: Any) -> Any
      :abstractmethod:

      Apply list of transforms to onnx model.

      Args:
          model: onnx model to be quantized.
      Returns:
          New onnx model based on `model` which has been transformed.



