:py:mod:`quark.onnx.utils.model_utils`
======================================

.. py:module:: quark.onnx.utils.model_utils

.. autoapi-nested-parse::

   Utility functions.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.utils.model_utils.get_tensor_value
   quark.onnx.utils.model_utils.generate_initializer
   quark.onnx.utils.model_utils.save_model



.. py:function:: get_tensor_value(initializer: onnx.TensorProto) -> numpy.ndarray[Any, numpy.dtype[numpy.float32]]

   Convert TensorProto to numpy array.


.. py:function:: generate_initializer(tensor_array: numpy.ndarray[Any, numpy.dtype[numpy.float32]], dtype: Any, name: str) -> onnx.TensorProto

   Generate initializers from numpy array.


.. py:function:: save_model(model: onnx.ModelProto, path: str, as_text: bool = False) -> None

   Save onnx model to disk.


