:py:mod:`quark.onnx.tools.convert_lstm_to_customlstm`
=====================================================

.. py:module:: quark.onnx.tools.convert_lstm_to_customlstm

.. autoapi-nested-parse::

   Convert Custom QDQ to QDQ.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.convert_lstm_to_customlstm.convert_lstm_to_customlstm
   quark.onnx.tools.convert_lstm_to_customlstm.custom_ops_infer_shapes



.. py:function:: convert_lstm_to_customlstm(model: onnx.ModelProto) -> Any

   Convert Custom LSTM to LSTM.
   :param model: source model
   :return: converted model


.. py:function:: custom_ops_infer_shapes(model: onnx.ModelProto) -> Any

   Generate value info for output tensors of custom ops.
   :param model: source model
   :return: converted model


