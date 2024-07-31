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

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization.
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range.
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights.
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param extra_options: Additional options for quantization.
   :type extra_options: Any, optional

   Inherits from:
       OrtQDQQuantizer: Base class for ONNX QDQ quantization.


.. py:class:: QDQNPUTransformerQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], extra_options: Optional[Dict[str, Any]] = None)




   A class to perform quantization on an ONNX model using Quantize-Dequantize (QDQ) nodes
   optimized for NPU (Neural Processing Unit) Transformers.

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization.
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range.
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights.
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param extra_options: Additional options for quantization.
   :type extra_options: Optional[Dict[str, Any]], optional

   Inherits from:
       QDQQuantizer: Base class for ONNX QDQ quantization.


.. py:class:: VitisQDQQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Any = None)




   A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization on an ONNX model.

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization.
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range.
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights.
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param calibrate_method: The method used for calibration.
   :type calibrate_method: Any
   :param quantized_tensor_type: Dictionary specifying quantized tensor types.
   :type quantized_tensor_type: Dict[Any, Any], optional
   :param extra_options: Additional options for quantization.
   :type extra_options: Any, optional

   Inherits from:
       VitisONNXQuantizer: Base class for Vitis-specific ONNX quantization.

   .. attribute:: tensors_to_quantize

      Dictionary of tensors to be quantized.

      :type: Dict[Any, Any]

   .. attribute:: bias_to_quantize

      List of bias tensors to be quantized.

      :type: List[Any]

   .. attribute:: nodes_to_remove

      List of nodes to be removed during quantization.

      :type: List[Any]

   .. attribute:: op_types_to_exclude_output_quantization

      List of op types to exclude from output quantization.

      :type: List[str]

   .. attribute:: quantize_bias

      Whether to quantize bias tensors.

      :type: bool

   .. attribute:: add_qdq_pair_to_weight

      Whether to add QDQ pairs to weights.

      :type: bool

   .. attribute:: dedicated_qdq_pair

      Whether to create dedicated QDQ pairs for each node.

      :type: bool

   .. attribute:: tensor_to_its_receiving_nodes

      Dictionary mapping tensors to their receiving nodes.

      :type: Dict[Any, Any]

   .. attribute:: qdq_op_type_per_channel_support_to_axis

      Dictionary mapping op types to channel axis for per-channel quantization.

      :type: Dict[str, int]

   .. attribute:: int32_bias

      Whether to quantize bias using int32.

      :type: bool

   .. attribute:: weights_only

      Whether to perform weights-only quantization.

      :type: bool


.. py:class:: VitisQDQNPUCNNQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   A class to perform Vitis-specific Quantize-Dequantize (QDQ) quantization for NPU (Neural Processing Unit) on CNN models.

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization (must be False for NPU).
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range (must be False for NPU).
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights (must be QuantType.QInt8 for NPU).
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param calibrate_method: The method used for calibration.
   :type calibrate_method: Any
   :param quantized_tensor_type: Dictionary specifying quantized tensor types.
   :type quantized_tensor_type: Dict[Any, Any], optional
   :param extra_options: Additional options for quantization.
   :type extra_options: Optional[Dict[str, Any]], optional

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   .. attribute:: tensors_to_quantize

      Dictionary of tensors to be quantized.

      :type: Dict[Any, Any]

   .. attribute:: is_weight_symmetric

      Whether to enforce symmetric quantization for weights.

      :type: bool

   .. attribute:: is_activation_symmetric

      Whether to enforce symmetric quantization for activations.

      :type: bool


.. py:class:: VitisExtendedQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, quant_format: Any, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any], extra_options: Optional[Dict[str, Any]] = None)




   A class to perform extended Vitis-specific Quantize-Dequantize (QDQ) quantization.

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization.
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range.
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param quant_format: The format for quantization.
   :type quant_format: Any
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights.
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param calibrate_method: The method used for calibration.
   :type calibrate_method: Any
   :param quantized_tensor_type: Dictionary specifying quantized tensor types.
   :type quantized_tensor_type: Dict[Any, Any]
   :param extra_options: Additional options for quantization.
   :type extra_options: Optional[Dict[str, Any]], optional

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   .. attribute:: tensors_to_quantize

      Dictionary of tensors to be quantized.

      :type: Dict[Any, Any]

   .. attribute:: quant_format

      The format for quantization.

      :type: Any

   .. attribute:: add_qdq_pair_to_weight

      Whether to add QDQ pair to weight (and bias).

      :type: bool

   .. attribute:: fold_relu

      Whether to fold ReLU layers.

      :type: bool


.. py:class:: VitisBFPQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   A class to perform Vitis-specific Block Floating Point (BFP) Quantization-Dequantization (QDQ) quantization.

   :param model: The ONNX model to be quantized.
   :type model: ModelProto
   :param per_channel: Whether to perform per-channel quantization.
   :type per_channel: bool
   :param reduce_range: Whether to reduce the quantization range.
   :type reduce_range: bool
   :param mode: The quantization mode to be used.
   :type mode: QuantizationMode.QLinearOps
   :param static: Whether to use static quantization.
   :type static: bool
   :param weight_qType: The quantization type for weights.
   :type weight_qType: Any
   :param activation_qType: The quantization type for activations.
   :type activation_qType: Any
   :param tensors_range: Dictionary specifying the min and max values for tensors.
   :type tensors_range: Any
   :param nodes_to_quantize: List of node names to be quantized.
   :type nodes_to_quantize: List[str]
   :param nodes_to_exclude: List of node names to be excluded from quantization.
   :type nodes_to_exclude: List[str]
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param calibrate_method: The method used for calibration.
   :type calibrate_method: Any
   :param quantized_tensor_type: Dictionary specifying quantized tensor types.
   :type quantized_tensor_type: Dict[Any, Any], optional
   :param extra_options: Additional options for quantization.
   :type extra_options: Optional[Dict[str, Any]], optional

   Inherits from:
       VitisQDQQuantizer: Base class for Vitis-specific QDQ quantization.

   .. attribute:: int32_bias

      Whether to quantize bias as int32.

      :type: bool

   .. attribute:: is_activation_symmetric

      Whether to use symmetric quantization for activations.

      :type: bool

   .. attribute:: bfp_attrs

      Attributes for BFP quantization.

      :type: Dict[str, Any]


