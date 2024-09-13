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

   .. py:class:: NodeType(*args, **kwds)




      Create a collection of name/value pairs.

      Example enumeration:

      >>> class Color(Enum):
      ...     RED = 1
      ...     BLUE = 2
      ...     GREEN = 3

      Access them by:

      - attribute access:

        >>> Color.RED
        <Color.RED: 1>

      - value lookup:

        >>> Color(1)
        <Color.RED: 1>

      - name lookup:

        >>> Color['RED']
        <Color.RED: 1>

      Enumerations can be iterated over, and know how many members they have:

      >>> len(Color)
      3

      >>> list(Color)
      [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

      Methods can be added to enumerations, and members can have their own
      attributes -- see the documentation for details.


   .. py:method:: transform() -> Tuple[onnx.ModelProto, Dict[str, Any]]

      Transforms the Onnx model by applying all the specified transforms.

      This is the main entry point function used to apply the transformations to
      the Onnx model.

      Not suitable for multi-threaded use. Creates and manipulates internal state.

      Returns:
        (Onnx model after transformation, Updated node metadata map)



