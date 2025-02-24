:orphan:

:py:mod:`quark.onnx.qdq_quantizer`
==================================

.. py:module:: quark.onnx.qdq_quantizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.qdq_quantizer.QDQQuantizer
   quark.onnx.qdq_quantizer.QDQNPUTransformerQuantizer
   quark.onnx.qdq_quantizer.VitisQDQQuantizer
   quark.onnx.qdq_quantizer.VitisQDQNPUCNNQuantizer
   quark.onnx.qdq_quantizer.VitisExtendedQuantizer
   quark.onnx.qdq_quantizer.VitisBFPQuantizer




.. py:class:: QDQQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], extra_options: Any = None)




   A class to perform quantization on an ONNX model using Quantize-Dequantize (QDQ) nodes.

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
       extra_options (Any, optional): Additional options for quantization.

   Inherits from:
       OrtQDQQuantizer: Base class for ONNX QDQ quantization.

   .. py:method:: quantize_bias_tensor(bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None

      Adds a bias tensor to the list of bias tensors to quantize. Called by op quantizers that
      want to quantize a bias with bias_zero_point = 0 and bias_scale = input_scale * weight_scale * beta.
      TODO: Explain the reasoning for using this formula.

      Args:
          node_name: name of the node that consumes the bias, input, and weight tensors.
          bias_name: name of the bias tensor to quantize.
          input_name: name of the input tensor whose scale is used to compute the bias's scale.
          weight_name: name of the weight tensor whose scale is used to compute the bias's scale.
          beta: Multiplier used to compute the bias's scale.



.. py:class:: QDQNPUTransformerQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], extra_options: Optional[Dict[str, Any]] = None)




   A class to perform quantization on an ONNX model using Quantize-Dequantize (QDQ) nodes
   optimized for NPU (Neural Processing Unit) Transformers.

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
       extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

   Inherits from:
       QDQQuantizer: Base class for ONNX QDQ quantization.

   .. py:method:: quantize_bias_tensor(bias_name: str, input_name: str, weight_name: str, beta: float = 1.0) -> None

      Adds a bias tensor to the list of bias tensors to quantize. Called by op quantizers that
      want to quantize a bias with bias_zero_point = 0 and bias_scale = input_scale * weight_scale * beta.
      TODO: Explain the reasoning for using this formula.

      Args:
          node_name: name of the node that consumes the bias, input, and weight tensors.
          bias_name: name of the bias tensor to quantize.
          input_name: name of the input tensor whose scale is used to compute the bias's scale.
          weight_name: name of the weight tensor whose scale is used to compute the bias's scale.
          beta: Multiplier used to compute the bias's scale.



.. py:class:: VitisQDQQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Any = None)




   A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization on an ONNX model.

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
       calibrate_method (Any): The method used for calibration.
       quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
       extra_options (Any, optional): Additional options for quantization.

   Inherits from:
       VitisONNXQuantizer: Base class for Vitis-specific ONNX quantization.

   Attributes:
       tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
       bias_to_quantize (List[Any]): List of bias tensors to be quantized.
       nodes_to_remove (List[Any]): List of nodes to be removed during quantization.
       op_types_to_exclude_output_quantization (List[str]): List of op types to exclude from output quantization.
       quantize_bias (bool): Whether to quantize bias tensors.
       add_qdq_pair_to_weight (bool): Whether to add QDQ pairs to weights.
       dedicated_qdq_pair (bool): Whether to create dedicated QDQ pairs for each node.
       tensor_to_its_receiving_nodes (Dict[Any, Any]): Dictionary mapping tensors to their receiving nodes.
       qdq_op_type_per_channel_support_to_axis (Dict[str, int]): Dictionary mapping op types to channel axis for per-channel quantization.
       int32_bias (bool): Whether to quantize bias using int32.
       weights_only (bool): Whether to perform weights-only quantization.



.. py:class:: VitisQDQNPUCNNQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization for NPU (Neural Processing Unit) on CNN models.

   Args:
       model (ModelProto): The ONNX model to be quantized.
       per_channel (bool): Whether to perform per-channel quantization (must be False for NPU).
       reduce_range (bool): Whether to reduce the quantization range (must be False for NPU).
       mode (QuantizationMode.QLinearOps): The quantization mode to be used.
       static (bool): Whether to use static quantization.
       weight_qType (Any): The quantization type for weights (must be QuantType.QInt8 for NPU).
       activation_qType (Any): The quantization type for activations.
       tensors_range (Any): Dictionary specifying the min and max values for tensors.
       nodes_to_quantize (List[str]): List of node names to be quantized.
       nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
       op_types_to_quantize (List[str]): List of operation types to be quantized.
       calibrate_method (Any): The method used for calibration.
       quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
       extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   Attributes:
       tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
       is_weight_symmetric (bool): Whether to enforce symmetric quantization for weights.
       is_activation_symmetric (bool): Whether to enforce symmetric quantization for activations.



.. py:class:: VitisExtendedQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, quant_format: Any, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any], extra_options: Optional[Dict[str, Any]] = None)




   A class to perform extended Vitis-specific Quantize-Dequantize (QDQ) quantization.

   Args:
       model (ModelProto): The ONNX model to be quantized.
       per_channel (bool): Whether to perform per-channel quantization.
       reduce_range (bool): Whether to reduce the quantization range.
       mode (QuantizationMode.QLinearOps): The quantization mode to be used.
       quant_format (Any): The format for quantization.
       static (bool): Whether to use static quantization.
       weight_qType (Any): The quantization type for weights.
       activation_qType (Any): The quantization type for activations.
       tensors_range (Any): Dictionary specifying the min and max values for tensors.
       nodes_to_quantize (List[str]): List of node names to be quantized.
       nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
       op_types_to_quantize (List[str]): List of operation types to be quantized.
       calibrate_method (Any): The method used for calibration.
       quantized_tensor_type (Dict[Any, Any]): Dictionary specifying quantized tensor types.
       extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   Attributes:
       tensors_to_quantize (Dict[Any, Any]): Dictionary of tensors to be quantized.
       quant_format (Any): The format for quantization.
       add_qdq_pair_to_weight (bool): Whether to add QDQ pair to weight (and bias).
       fold_relu (bool): Whether to fold ReLU layers.



.. py:class:: VitisBFPQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, quant_format: Any, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   A class to perform Vitis-specific Block Floating Point (BFP) Quantization-Dequantization (QDQ) quantization.

   Args:
       model (ModelProto): The ONNX model to be quantized.
       per_channel (bool): Whether to perform per-channel quantization.
       reduce_range (bool): Whether to reduce the quantization range.
       mode (QuantizationMode.QLinearOps): The quantization mode to be used.
       quant_format (Any): The format for quantization.
       static (bool): Whether to use static quantization.
       weight_qType (Any): The quantization type for weights.
       activation_qType (Any): The quantization type for activations.
       tensors_range (Any): Dictionary specifying the min and max values for tensors.
       nodes_to_quantize (List[str]): List of node names to be quantized.
       nodes_to_exclude (List[str]): List of node names to be excluded from quantization.
       op_types_to_quantize (List[str]): List of operation types to be quantized.
       calibrate_method (Any): The method used for calibration.
       quantized_tensor_type (Dict[Any, Any], optional): Dictionary specifying quantized tensor types.
       extra_options (Optional[Dict[str, Any]], optional): Additional options for quantization.

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   Attributes:
       int32_bias (bool): Whether to quantize bias as int32.
       is_activation_symmetric (bool): Whether to use symmetric quantization for activations.
       quant_format (Any): The format for quantization.
       fn_type: (string): The op type of the fix neuron.
       fn_attrs (Dict[str, Any]): Attributes for BFP/MX fix neuron.


