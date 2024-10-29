:py:mod:`quark.onnx.tools.remove_qdq_mul_add`
=============================================

.. py:module:: quark.onnx.tools.remove_qdq_mul_add

.. autoapi-nested-parse::

   Remove QDQ in the `mul + q + dq + add` structure operators.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.remove_qdq_mul_add.remove_qdq_mul_add
   quark.onnx.tools.remove_qdq_mul_add.find_node_by_output



.. py:function:: remove_qdq_mul_add(onnx_model: onnx.ModelProto) -> Any

   Modify an ONNX quantized model to remove q and dq ops in the `mul + q + dq + add` structure.
   Start from `Add` nodes and traverse upwards.

   :param onnx_model: The input ONNX model.
   :return: Modified ONNX model with q and dq ops in the `mul + q + dq + add` structure removed.


.. py:function:: find_node_by_output(nodes: List[onnx.NodeProto], output_name: str) -> Optional[onnx.NodeProto]

   Find a node that produces the specified output.

   :param nodes: List of nodes to search.
   :param output_name: The output name to match.
   :return: The node that matches the output or None if not found.


