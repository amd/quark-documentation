:orphan:

:py:mod:`quark.onnx.operators.custom_ops.test_custom_ops_qdq`
=============================================================

.. py:module:: quark.onnx.operators.custom_ops.test_custom_ops_qdq


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.operators.custom_ops.test_custom_ops_qdq.test_custom_op



.. py:function:: test_custom_op(data_type: onnx.onnx_ml_pb2.TensorProto.DataType, scale: Any, zero_point: Any, quant_type: Any) -> None

   y_scale_initializer = onnx.numpy_helper.from_array(scale, name="y_scale")
   y_zp_initializer = onnx.numpy_helper.from_array(zero_point,
                                                   name="y_zero_point")
   x_scale_initializer = onnx.numpy_helper.from_array(scale, name="x_scale")
   x_zp_initializer = onnx.numpy_helper.from_array(zero_point,
                                                   name="x_zero_point")
   initializers = [
       y_scale_initializer, y_zp_initializer, x_scale_initializer,
       x_zp_initializer
   ]


