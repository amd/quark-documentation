:orphan:

:py:mod:`quark.onnx.tools.float16`
==================================

.. py:module:: quark.onnx.tools.float16


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.float16.convert_np_to_float16
   quark.onnx.tools.float16.convert_tensor_float_to_float16
   quark.onnx.tools.float16.convert_float_to_float16
   quark.onnx.tools.float16.convert_float_to_float16_model_path
   quark.onnx.tools.float16.convert_np_to_float
   quark.onnx.tools.float16.convert_tensor_float16_to_float
   quark.onnx.tools.float16.convert_float16_to_float



.. py:function:: convert_np_to_float16(np_array: numpy.typing.NDArray[numpy.float32], min_positive_val: float = 1e-07, max_finite_val: float = 10000.0) -> numpy.typing.NDArray[numpy.float16]

   Convert float32 numpy array to float16 without changing sign or finiteness.
   Positive values less than min_positive_val are mapped to min_positive_val.
   Positive finite values greater than max_finite_val are mapped to max_finite_val.
   Similar for negative values. NaN, 0, inf, and -inf are unchanged.


.. py:function:: convert_tensor_float_to_float16(tensor: onnx.onnx_pb.TensorProto, min_positive_val: float = 1e-07, max_finite_val: float = 10000.0) -> onnx.onnx_pb.TensorProto

   Convert tensor float to float16.

   :param tensor: TensorProto object
   :return tensor_float16: converted TensorProto object

   Example:

   ::

       from onnxmltools.utils.float16_converter import convert_tensor_float_to_float16
       new_tensor = convert_tensor_float_to_float16(tensor)



.. py:function:: convert_float_to_float16(model: onnx.onnx_pb.ModelProto, min_positive_val: float = 1e-07, max_finite_val: float = 10000.0, keep_io_types: bool = False, disable_shape_infer: bool = False, op_block_list: Union[List[str], None] = None, node_block_list: Union[List[str], None] = None) -> onnx.onnx_pb.ModelProto

   Convert tensor float type in the ONNX ModelProto input to tensor float16.

   :param model: ONNX ModelProto object
   :param disable_shape_infer: Type/shape information is needed for conversion to work.
                               Set to True only if the model already has type/shape information for all tensors.
   :return: converted ONNX ModelProto object

   Examples:

   ::

       Example 1: Convert ONNX ModelProto object:
       import float16
       new_onnx_model = float16.convert_float_to_float16(onnx_model)

       Example 2: Convert ONNX model binary file:
       import onnx
       import float16
       onnx_model = onnx.load_model('model.onnx')
       new_onnx_model = float16.convert_float_to_float16(onnx_model)
       onnx.save_model(new_onnx_model, 'new_model.onnx')



.. py:function:: convert_float_to_float16_model_path(model_path: str, min_positive_val: float = 1e-07, max_finite_val: float = 10000.0, keep_io_types: bool = False) -> onnx.onnx_pb.ModelProto

   Convert tensor float type in the ONNX Model to tensor float16.
   *It is to fix an issue that infer_shapes func cannot be used to infer >2GB models.
   *But this function can be applied to all model sizes.
   :param model_path: ONNX Model path
   :return: converted ONNX ModelProto object
   Examples
   ::
       #Convert to ONNX ModelProto object and save model binary file:
       from onnxmltools.utils.float16_converter import convert_float_to_float16_model_path
       new_onnx_model = convert_float_to_float16_model_path('model.onnx')
       onnx.save(new_onnx_model, 'new_model.onnx')


.. py:function:: convert_np_to_float(np_array: numpy.typing.NDArray[numpy.float16], min_positive_val: float = 1e-07, max_finite_val: float = 10000.0) -> numpy.typing.NDArray[numpy.float32]

   Convert float16 numpy array to float32 without changing sign or finiteness.
   Similar for negative values. NaN, 0, inf, and -inf are unchanged.


.. py:function:: convert_tensor_float16_to_float(tensor: onnx.onnx_pb.TensorProto) -> onnx.onnx_pb.TensorProto

   Convert tensor float16 to float.

   :param tensor: TensorProto object
   :return tensor_float: converted TensorProto object

   Example:

   ::

       new_tensor = convert_tensor_float16_to_float(tensor)



.. py:function:: convert_float16_to_float(model: onnx.onnx_pb.ModelProto, disable_shape_infer: bool = False, op_block_list: Optional[List[str]] = None, node_block_list: Optional[List[str]] = None) -> onnx.onnx_pb.ModelProto

   Convert tensor float16 type in the ONNX ModelProto input to tensor float.

   :param model: ONNX ModelProto object
   :param disable_shape_infer: Type/shape information is needed for conversion to work.
                               Set to True only if the model already has type/shape information for all tensors.
   :return: converted ONNX ModelProto object

   Examples:

   ::

       Example 1: Convert ONNX ModelProto object:
       import float16
       new_onnx_model = float16.convert_float16_to_float(onnx_model)

       Example 2: Convert ONNX model binary file:
       import onnx
       import float16
       onnx_model = onnx.load_model('model.onnx')
       new_onnx_model = float16.convert_float16_to_float(onnx_model)
       onnx.save_model(new_onnx_model, 'new_model.onnx')



