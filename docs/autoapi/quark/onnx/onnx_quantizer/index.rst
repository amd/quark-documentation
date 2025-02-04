:orphan:

:py:mod:`quark.onnx.onnx_quantizer`
===================================

.. py:module:: quark.onnx.onnx_quantizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.onnx_quantizer.ONNXQuantizer
   quark.onnx.onnx_quantizer.VitisONNXQuantizer




.. py:class:: ONNXQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], extra_options: Optional[Dict[str, Any]] = None)




   A class to perform quantization on an ONNX model.

   Args:
       model (ModelProto): The ONNX model to be quantized.
       per_channel (bool): Whether to perform per-channel quantization.
       reduce_range (bool): Whether to reduce the quantization range.
       mode (QuantizationMode.QLinearOps): The quantization mode to be used.
       static (bool): Whether to use static quantization.
       weight_qType (Any): The quantization type for weights.
       activation_qType (Any): The quantization type for activations.
       tensors_range (Any): The range of tensors for quantization.
       nodes_to_quantize (List[str]): List of node names to be quantized.
       nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
       op_types_to_quantize (List[str]): List of operation types to be quantized.
       extra_options (Optional[Dict[str, Any]]): Additional options for quantization.

   Inherits from:
       OrtONNXQuantizer: Base class for ONNX quantization.


.. py:class:: VitisONNXQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   A class to perform quantization on an ONNX model specifically optimized for Vitis AI.

   Args:
       model (ModelProto): The ONNX model to be quantized.
       per_channel (bool): Whether to perform per-channel quantization.
       reduce_range (bool): Whether to reduce the quantization range.
       mode (QuantizationMode.QLinearOps): The quantization mode to be used.
       static (bool): Whether to use static quantization.
       weight_qType (Any): The quantization type for weights.
       activation_qType (Any): The quantization type for activations.
       tensors_range (Any): Dictionary specifying the min and max values for tensors.
       nodes_to_quantize (List[str]): List of node names to be quantized.
       nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
       op_types_to_quantize (List[str]): List of operation types to be quantized.
       calibrate_method (Any): The calibration method to be used.
       quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying the types for quantized tensors.
       extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

   Inherits from:
       OrtONNXQuantizer: Base class for ONNX quantization.

   .. py:method:: find_quant_scale_zp(input_name: str) -> Any

      Finds the quantization scale and zero-point for a given input.

      This method looks up the quantization scale and zero-point values for the specified input name.
      It first checks the current instance's `used_scale_zp_map`. If not found, it recursively checks
      the parent instance if one exists.

      :param input_name: The name of the input for which to find the quantization scale and zero-point.
      :type input_name: str
      :return: A tuple containing the quantization scale and zero-point if found, otherwise (None, None).
      :rtype: Any


   .. py:method:: find_quantized_value(input_name: str) -> Any

      Finds the quantized value for a given input.

      This method looks up the quantized value for the specified input name.
      It first checks the current instance's `quantized_value_map`. If not found, it recursively checks
      the parent instance if one exists.

      :param input_name: The name of the input for which to find the quantized value.
      :type input_name: str
      :return: The quantized value if found, otherwise None.
      :rtype: Any


   .. py:method:: quantize_bias_static(bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> Any

      Quantizes the bias using static quantization. Zero Point == 0 and Scale == Input_Scale * Weight_Scale.

      This method performs the following steps:
      1. Validates the weight quantization type.
      2. Retrieves the scale for the weight.
      3. Retrieves the bias data and its scale.
      4. Retrieves the scale for the input.
      5. Calculates the scale for the bias.
      6. Quantizes the bias data.
      7. Updates the bias, scale, and zero-point initializers in the model.
      8. Updates the quantized value map with the new quantized bias information.

      :param bias_name: The name of the bias to be quantized.
      :type bias_name: str
      :param input_name: The name of the input associated with the bias.
      :type input_name: str
      :param weight_name: The name of the weight associated with the bias.
      :type weight_name: str
      :param beta: A scaling factor applied during quantization. Default is 1.0.
      :type beta: float
      :return: The name of the quantized bias.
      :rtype: Any

      :raises ValueError: If the weight quantization type is not supported or if the input name is not found in the quantized value map.


   .. py:method:: quantize_weight(node: onnx.NodeProto, indices: Any, reduce_range: bool = False, op_level_per_channel: bool = False, axis: int = -1, from_subgraph: bool = False) -> Any

      Quantizes the weights of a given node.

      In some circumstances, a weight is not an initializer. For example, in MatMul, if both A and B are not initializers,
      B can still be considered as a weight.

      This method calls `__quantize_inputs` to perform the weight quantization.

      :param node: The node containing the weights to be quantized.
      :type node: NodeProto
      :param indices: The indices of the inputs to be quantized.
      :type indices: Any
      :param reduce_range: Flag to indicate whether to reduce the quantization range. Default is False.
      :type reduce_range: bool, optional
      :param op_level_per_channel: Flag to indicate whether to use per-channel quantization at the operator level. Default is False.
      :type op_level_per_channel: bool, optional
      :param axis: The axis for per-channel quantization. Default is -1.
      :type axis: int, optional
      :param from_subgraph: Flag to indicate whether the node is from a subgraph. Default is False.
      :type from_subgraph: bool, optional
      :return: The result of the weight quantization process.
      :rtype: Any


   .. py:method:: quantize_initializer(weight: Any, qType: Any, method: Any, reduce_range: bool = False, keep_float_weight: bool = False) -> Tuple[str, str, str]

      :param weight: TensorProto initializer
      :param qType: type to quantize to. Note that it may be different with weight_qType because of mixed precision
      :param keep_float_weight: Whether to quantize the weight. In some cases, we only want to qunatize scale and zero point.
                                If keep_float_weight is False, quantize the weight, or don't quantize the weight.
      :return: quantized weight name, zero point name, scale name


   .. py:method:: quantize_weight_per_channel(weight_name: str, weight_qType: Any, channel_axis: Any, method: Any, reduce_range: bool = True, keep_float_weight: bool = False) -> Tuple[str, str, str]

      Quantizes the given weight tensor per channel.

      This method quantizes the weights per channel, creating separate quantization parameters (scale and zero-point) for each channel.

      :param weight_name: The name of the weight tensor to be quantized.
      :type weight_name: str
      :param weight_qType: The data type to use for quantization.
      :type weight_qType: Any
      :param channel_axis: The axis representing the channel dimension in the weight tensor.
      :type channel_axis: Any
      :param method: The quantization method to use.
      :type method: Any
      :param reduce_range: Whether to reduce the quantization range. Default is True.
      :type reduce_range: bool, optional
      :param keep_float_weight: Whether to keep the original floating-point weights. Default is False.
      :type keep_float_weight: bool, optional
      :return: A tuple containing the names of the quantized weight tensor, zero-point tensor, and scale tensor.
      :rtype: Tuple[str, str, str]

      :raises ValueError: If the specified weight is not an initializer.


   .. py:method:: calculate_quantization_params() -> Any

      Calculates the quantization parameters for each tensor in the model.

      This method computes the quantization parameters (scale and zero-point) for each tensor in the model
      based on its range (rmin and rmax). It adjusts the tensor ranges for the inputs of Clip and Relu nodes
      and ensures the correct quantization parameters are used for each tensor type.

      :return: A dictionary containing the quantization parameters for each tensor.
      :rtype: Any

      :raises ValueError: If a weight is not an initializer.

      Notes:
          - If `self.tensors_range` is None, the method returns immediately.
          - Adjusts tensor ranges for Clip and Relu nodes.
          - For versions of ONNX Runtime below 1.16.0, specific quantization parameters are computed.
          - For versions of ONNX Runtime 1.16.0 and above, the `QuantizationParams` class is used.
          - Forces asymmetric quantization for ReLU-like output tensors if `self.use_unsigned_relu` is True.



