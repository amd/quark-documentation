:py:mod:`quark.onnx.tools.convert_customqdq_to_qdq`
===================================================

.. py:module:: quark.onnx.tools.convert_customqdq_to_qdq

.. autoapi-nested-parse::

   Convert Custom QDQ to QDQ.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.convert_customqdq_to_qdq.convert_customqdq_to_qdq
   quark.onnx.tools.convert_customqdq_to_qdq.custom_ops_infer_shapes



.. py:function:: convert_customqdq_to_qdq(model: onnx.ModelProto) -> Any

   Convert Custom QDQ to Standard QDQ.
   :param model: source model
   :return: converted model


.. py:function:: custom_ops_infer_shapes(model: onnx.ModelProto) -> Any

   Generate value info for output tensors of custom ops.
   :param model: source model
   :return: converted model


