:py:mod:`quark.onnx.tools.remove_qdq_between_ops`
=================================================

.. py:module:: quark.onnx.tools.remove_qdq_between_ops

.. autoapi-nested-parse::

   Remove QuantizeLinear (q) and DequantizeLinear (dq) nodes between specified operator pairs.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.remove_qdq_between_ops.remove_qdq_between_ops
   quark.onnx.tools.remove_qdq_between_ops.find_node_by_output



.. py:function:: remove_qdq_between_ops(model: onnx.ModelProto, between_ops: Union[list[Tuple[str, str]], Any]) -> Any

   Modify an ONNX quantized model to remove q and dq ops between specified operation pairs.
   Start from `lower_op` nodes and traverse upwards to `upper_op`.

   :param model: The input ONNX model to be modified.
   :param between_ops: A list of tuples where each tuple contains two operator types.
                       The function will look for `q` and `dq` nodes between these pairs.
   :return: The modified ONNX model with the specified q and dq nodes removed.


.. py:function:: find_node_by_output(nodes: List[onnx.NodeProto], output_name: str) -> Optional[onnx.NodeProto]

   Find a node that produces the specified output.

   :param nodes: List of nodes to search.
   :param output_name: The output name to match.
   :return: The node that matches the output or None if not found.


