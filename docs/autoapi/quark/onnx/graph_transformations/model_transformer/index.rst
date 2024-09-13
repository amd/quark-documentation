:py:mod:`quark.onnx.graph_transformations.model_transformer`
============================================================

.. py:module:: quark.onnx.graph_transformations.model_transformer

.. autoapi-nested-parse::

   Apply graph transformations to a onnx model.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.graph_transformations.model_transformer.ModelTransformer




.. py:class:: ModelTransformer(model: onnx.ModelProto, transforms: List[Any], candidate_nodes: Optional[Dict[str, Any]] = None, node_metadata: Optional[Dict[str, Any]] = None)




   Matches patterns to apply transforms in a tf.keras model graph.

   .. py:class:: NodeType




      Generic enumeration.

      Derive from this class to define new enumerations.


   .. py:method:: transform() -> Tuple[onnx.ModelProto, Dict[str, Any]]

      Transforms the Onnx model by applying all the specified transforms.

      This is the main entry point function used to apply the transformations to
      the Onnx model.

      Not suitable for multi-threaded use. Creates and manipulates internal state.

      Returns:
        (Onnx model after transformation, Updated node metadata map)



