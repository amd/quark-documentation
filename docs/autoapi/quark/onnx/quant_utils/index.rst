:orphan:

:py:mod:`quark.onnx.quant_utils`
================================

.. py:module:: quark.onnx.quant_utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.quant_utils.Int16Method
   quark.onnx.quant_utils.PowerOfTwoMethod
   quark.onnx.quant_utils.VitisQuantType
   quark.onnx.quant_utils.VitisQuantFormat
   quark.onnx.quant_utils.CachedDataReader
   quark.onnx.quant_utils.RandomDataReader
   quark.onnx.quant_utils.PathDataReader



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.quant_utils.is_ort_version_below
   quark.onnx.quant_utils.get_qmin_qmax_for_qType
   quark.onnx.quant_utils.get_qrange_for_qType
   quark.onnx.quant_utils.infer_shape
   quark.onnx.quant_utils.get_datatype_shape
   quark.onnx.quant_utils.dump_model
   quark.onnx.quant_utils.is_approximately_equal
   quark.onnx.quant_utils.check_reduce_mean_condition
   quark.onnx.quant_utils.check_hard_sigmoid_condition
   quark.onnx.quant_utils.is_leaky_relu_with_alpha
   quark.onnx.quant_utils.is_clip_with_min_max
   quark.onnx.quant_utils.is_node_needs_annotated
   quark.onnx.quant_utils.get_annotate_tensors
   quark.onnx.quant_utils.get_qdq_to_remove
   quark.onnx.quant_utils.customqdq_to_contribqdq
   quark.onnx.quant_utils.remove_nodes
   quark.onnx.quant_utils.remove_initializers
   quark.onnx.quant_utils.modified_annotate_input
   quark.onnx.quant_utils.scale2pos
   quark.onnx.quant_utils.pos2scale
   quark.onnx.quant_utils.compute_scale_zp
   quark.onnx.quant_utils.compute_scale_zp_fp
   quark.onnx.quant_utils.dequantize_data
   quark.onnx.quant_utils.quantize_data_pof2s
   quark.onnx.quant_utils.get_exclude_nodes
   quark.onnx.quant_utils.run_onnx_model
   quark.onnx.quant_utils.check_onnx_model
   quark.onnx.quant_utils.check_model_quantizable
   quark.onnx.quant_utils.dpu_leaky_relu_alpha
   quark.onnx.quant_utils.get_clip_min_max
   quark.onnx.quant_utils.check_relu_like_node
   quark.onnx.quant_utils.print_quantize_info
   quark.onnx.quant_utils.print_quantize_dynamic_info
   quark.onnx.quant_utils.find_int16_scale



.. py:function:: is_ort_version_below(target_version: str) -> bool

   This function checks whether the current version of ONNX Runtime (ORT) is below a specified version.

   Args:
       target_version (str): The version to compare against the current ORT version.

   Returns:
       True if the current ORT version is less than the target version, False otherwise.


.. py:class:: Int16Method(*args, **kwds)




   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

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


.. py:class:: PowerOfTwoMethod(*args, **kwds)




   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

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


.. py:class:: VitisQuantType(*args, **kwds)




   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

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


.. py:class:: VitisQuantFormat(*args, **kwds)




   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access::

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


.. py:function:: get_qmin_qmax_for_qType(qType: int, reduce_range: bool = False, symmetric: bool = False) -> Any

   Return qmin and qmax, the minimum and maximum value representable by the given qType
   :parameter qType: onnx.onnx_pb.TensorProto.UINT8/16 or onnx.onnx_pb.TensorProto.UINT8/16
   :return: qmin, qmax


.. py:function:: get_qrange_for_qType(qType: int, reduce_range: bool = False, symmetric: bool = False) -> Any

   Helper function to get the quantization range for a type.
       parameter qType: quantization type.
       return: quantization range.


.. py:class:: CachedDataReader(dr: onnxruntime.quantization.calibrate.CalibrationDataReader, data_size: Optional[int] = None, convert_nchw_to_nhwc: bool = False)




   A CalibrationDataReader cached input data from the user provided data reader.

   .. py:method:: reset_iter() -> None

      Recreate the iter so that it can iterate again


   .. py:method:: get_next() -> Optional[Dict[str, numpy.ndarray[Any, Any]]]

      Get next feed data
      :return: feed dict for the model



.. py:class:: RandomDataReader(model_path: str, input_shape: Union[List[int], Tuple[int], List[List[int]], Dict[str, List[int]], List[Any], Any] = [], input_data_range: Optional[str] = None)




   A CalibrationDataReader using random data for rapid quantiation.

   .. py:method:: get_next() -> Optional[Dict[str, numpy.ndarray[Any, Any]]]

      Get next feed data
      :return: feed dict for the model



.. py:class:: PathDataReader(model_path: str, data_path: str, input_shape: List[Any] = [])




   A CalibrationDataReader loading data from specified paths for model calibration.

   .. py:method:: get_next() -> Optional[Dict[str, numpy.ndarray[Any, Any]]]

      Get next feed data
      :return: feed dict for the model



.. py:function:: infer_shape(model: onnx.onnx_ml_pb2.ModelProto) -> onnx.onnx_ml_pb2.ModelProto

   :param model: the source model
   :return: the target model contains inferred shape


.. py:function:: get_datatype_shape(tensor: onnx.onnx_ml_pb2.TensorProto) -> Tuple[str, List[Any]]

   :param tensor: the input tensor
   :return: datatype and shape of the tensor


.. py:function:: dump_model(model: Union[str, onnx.ModelProto], dump_data_reader: Optional[object] = None, random_data_reader_input_shape: List[int] = [], dump_float: bool = False, output_dir: str = './dump_results') -> None

   This function dumps the simulation results of the quantized model,
   including weights and activation results.
   :param model: the input model
   :param dump_data_reader: data reader for dumpping
   :param random_data_reader_input_shape: if use internal random data reader,
          this is used to configure input node's shape
   :param dump_float: dump results of the float model or not
   :param output_dir: output directory for results


.. py:function:: is_approximately_equal(a: float, b: float, epsilon: float = 1e-06) -> bool

   :param a: scalar input
   :param b: scalar input
   :param epsilon: difference tolerance
   :return: equal or not


.. py:function:: check_reduce_mean_condition(model: onnx.ModelProto, node: onnx.NodeProto) -> bool

   Check conditions for Reduce Mean operation in ONNX graph nodes.

   :param model: ONNX model
   :param node: ONNX node
   :return: True if conditions for Reduce Mean are satisfied, False otherwise


.. py:function:: check_hard_sigmoid_condition(node: onnx.NodeProto) -> bool

   :param node: node object
   :return: hard sigmoid or not


.. py:function:: is_leaky_relu_with_alpha(node: onnx.NodeProto, alpha_value: float = 0.1) -> bool

   :param node: node object
   :param alpha_value: DPU supported alpha value
   :return: the Leaky ReLU node has a approximately alpha or not


.. py:function:: is_clip_with_min_max(model: onnx.ModelProto, node: onnx.NodeProto, min_value: float = 0.0, max_value: float = 6.0) -> bool

   :param model: model object
   :param node: node object
   :param min_value: supported minimum value of Clip
   :param max_value: supported maximum value of Clip
   :return: the Clip node has supported min and max value or not


.. py:function:: is_node_needs_annotated(model: onnx.ModelProto, node: onnx.NodeProto) -> bool

   :param model: model object
   :param node: node object
   :return: the node needs annotated or not


.. py:function:: get_annotate_tensors(model: onnx.ModelProto) -> List[str]

   Find patterns in the model where qdq needs to be removed, and then return the corresponding tensor names
   annotate_tensors refers to the tensors associated with the input of the qdq that need to be removed
   :param model: model object
   :return: the annotate tensors


.. py:function:: get_qdq_to_remove(model: onnx.ModelProto, relu_input: List[str]) -> Tuple[List[onnx.NodeProto], List[onnx.NodeProto], Dict[str, str]]

   Return the names of nodes to be removed and a dictionary for converting input tensors
   :param model: model object
   :param relu_input: the ReLU node inputs list
   :return: de-quantize & quantize nodes to remove and node mapping dict


.. py:function:: customqdq_to_contribqdq(model_path: str, use_external_data_format: bool) -> None

   Convert the custom QDQs to the contrib QDQs in the model
   :param model_path: the model path
   :return: None


.. py:function:: remove_nodes(model: onnx.ModelProto, nodes_list: List[Any]) -> onnx.ModelProto

   Delete nodes according to the nodes in the list
   :param model: model object
   :param nodes_list: nodes list to remove
   :return: the model that has removed some nodes


.. py:function:: remove_initializers(model: onnx.onnx_ml_pb2.ModelProto, init_list: List[str]) -> onnx.onnx_ml_pb2.ModelProto

   Delete initializers according to the initializer in the list
   :param model: model object
   :param init_list: initializer's name list to remove
   :return: the model that has removed some initializers


.. py:function:: modified_annotate_input(model: onnx.onnx_ml_pb2.ModelProto, input_node_mapping: Dict[str, str]) -> onnx.onnx_ml_pb2.ModelProto

   Modify the input of ReLU to the output of annotate op, and delete QDQ
   :param model: model object
   :param input_node_mapping: input node mapping dict
   :return: the modified model


.. py:function:: scale2pos(scale: float) -> int

   Obtain the fixed-point position corresponding to the scale.
   To avoid generating infinity during computations,
   the range of scale is limited.
   :param scale: the scale
   :return: the fixed-point position


.. py:function:: pos2scale(pos: int) -> float

   Obtain the scale corresponding to the fixed-point position.
   :param scale: the fixed-point position
   :return: the scale


.. py:function:: compute_scale_zp(rmin: numpy.ndarray[Any, Any], rmax: numpy.ndarray[Any, Any], qmin: numpy.ndarray[Any, Any], qmax: numpy.ndarray[Any, Any], element_type: int, method: PowerOfTwoMethod, symmetric: bool = False, use_pof2s: bool = True) -> Any

   Calculate the scale s and zero point z for the quantization relation
   r = s(q-z), where r are the original values and q are the corresponding
   quantized values.

   r and z are calculated such that every value within [rmin,rmax] has an
   approximate representation within [qmin,qmax]. In addition, qmin <= z <=
   qmax is enforced. If the symmetric flag is set to True, the interval
   [rmin,rmax] is symmetrized to [-absmax, +absmax], where
   absmax = max(abs(rmin), abs(rmax)).

   :parameter rmin: minimum value of r
   :parameter rmax: maximum value of r
   :parameter qmin: minimum value representable by the target quantization data type
   :parameter qmax: maximum value representable by the target quantization data type
   :return: zero and scale [z, s] of pof2s



.. py:function:: compute_scale_zp_fp(rmin: numpy.ndarray[Any, Any], rmax: numpy.ndarray[Any, Any], element_type: int, symmetric: bool = True) -> List[Any]

   Calculate the scale and zero point for a float type.

   :param rmin: minimum value of r
   :param rmax: maximum value of r
   :param element_type: the element data type of the tensor to quantize
   :return: zero and scale [z, s] of pof2s


.. py:function:: dequantize_data(data: numpy.ndarray[Any, Any], scale: numpy.ndarray[Any, Any], zero_point: numpy.ndarray[Any, Any]) -> Any

   :param data: the input data
   :param scale: the scale for quantization
   :param zero_point: the zero point for quantization
   :return: the de-quantized data


.. py:function:: quantize_data_pof2s(data: numpy.ndarray[Any, Any], qType: int, symmetric: bool, reduce_range: bool = False, rmin_real_range: Optional[float] = None, rmin_override: Optional[numpy.ndarray[Any, Any]] = None, rmax_override: Optional[numpy.ndarray[Any, Any]] = None, method: PowerOfTwoMethod = PowerOfTwoMethod.NonOverflow, pos_range: int = 5, use_pof2s: bool = True) -> Any

   :param data: data to quantize
   :param qType: data type to quantize to. Supported types UINT8/16 and INT8/16
   :param symmetric: whether symmetric quantization is used or not. This is applied to INT8/16.
   :return: minimum, maximum, zero point, scale, and quantized weights

   To pack weights, we compute a linear transformation

   - when data `type == uint8` mode, from `[rmin, rmax]` -> :math:`[0, 2^{b-1}]` and
   - when data `type == int8`, from `[-m , m]` -> :math:`[-(2^{b-1}-1), 2^{b-1}-1]` where
       `m = max(abs(rmin), abs(rmax))`

   and add necessary intermediate nodes to trasnform quantized weight to full weight using the equation

   :math:`r = S(q-z)`, where

   - *r*: real original value
   - *q*: quantized value
   - *S*: scale
   - *z*: zero point


.. py:function:: get_exclude_nodes(model_path: str, input_nodes: Union[List[str], None], output_nodes: Union[List[str], None]) -> List[str]

   Return the nodes to be excluded based on the given input and output nodes.
   :param model_path: the model path
   :param input_nodes: the nodes to start quantizing
   :param zero_point: the nodes to terminate quantizing
   :return: the nodes excluded from quantization


.. py:function:: run_onnx_model(model_path: str, data_reader: Any) -> None

   Check if the input ONNX can run successfully
   :param model_path: the model path
   :param data_reader: the data reader for feeding data


.. py:function:: check_onnx_model(model_path: str) -> None

   Check if the input ONNX can create InferenceSession successfully
   :param model_path: the model path


.. py:function:: check_model_quantizable(model: onnx.onnx_ml_pb2.ModelProto, op_types_to_quantize: Optional[List[str]], nodes_to_exclude: List[str]) -> bool

   Check if the model can be quantized.


.. py:function:: dpu_leaky_relu_alpha(x: float) -> float

   This function implements a DPU-specific Leaky ReLU activation with alpha value correction.


.. py:function:: get_clip_min_max(model: onnx.onnx_ml_pb2.ModelProto, clip_node: onnx.onnx_ml_pb2.NodeProto) -> Tuple[Optional[float], Optional[float], Optional[int]]

   Get clip min and max value from Clip node.
   :param model: onnx model instance
   :param clip_node: target Clip node
   :return: the min, max value and para type
            The meaning of para type is:
            None - unknown
            0 - attribute
            1 - initializer
            2 - other nodes


.. py:function:: check_relu_like_node(model: onnx.onnx_ml_pb2.ModelProto, node: onnx.onnx_ml_pb2.NodeProto) -> bool

   Check if the node is a relu-like node
   :param model: the model instance
   :param node: the node to check
   :return: True if it is


.. py:function:: print_quantize_info(model_input: str, model_output: str, calibration_data_reader: str, calibration_data_path: Union[str, None], quant_format: Union[Any, VitisQuantFormat], input_nodes: Union[List[str], None], output_nodes: Union[List[str], None], op_types_to_quantize: Union[List[str], None], random_data_reader_input_shape: Union[List[int], Tuple[int], List[List[int]], Dict[str, List[int]], List[Any], None], per_channel: bool, reduce_range: bool, activation_type: Union[Any, VitisQuantType], weight_type: Union[Any, VitisQuantType], nodes_to_quantize: List[str], nodes_to_exclude: List[str], optimize_model: bool, use_external_data_format: bool, calibrate_method: Union[Any, PowerOfTwoMethod, Int16Method], execution_providers: Union[List[str], None], enable_npu_cnn: bool, enable_npu_transformer: bool, specific_tensor_precision: bool, debug_mode: bool, convert_fp16_to_fp32: bool, convert_nchw_to_nhwc: bool, include_cle: bool, include_sq: bool, include_fast_ft: bool, extra_options: Dict[str, Any]) -> None

   print os_cpu, time, tool_version, quantized_configuration information.


.. py:function:: print_quantize_dynamic_info(model_input: Union[str, pathlib.Path, onnx.ModelProto], model_output: Union[str, pathlib.Path], op_types_to_quantize: Union[List[str], None], per_channel: bool, reduce_range: bool, weight_type: Union[Any, VitisQuantType], nodes_to_quantize: List[str], nodes_to_exclude: List[str], use_external_data_format: bool, debug_mode: bool, extra_options: Dict[str, Any]) -> None

   print os_cpu, time, tool_version, quantized_configuration information.


.. py:function:: find_int16_scale(x: float) -> Tuple[float, float, float]

   Given a float value, find the closest value corresponding to  M and 2**N,
   where the range of M and 2**N is within the representation range of int16 and uint16.


