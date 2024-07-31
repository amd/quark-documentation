:orphan:

:py:mod:`quark.onnx.cpu_quantizer`
==================================

.. py:module:: quark.onnx.cpu_quantizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.cpu_quantizer.VitisQDQCPUQuantizer




.. py:class:: VitisQDQCPUQuantizer(model: onnx.ModelProto, per_channel: bool, reduce_range: bool, mode: onnxruntime.quantization.quant_utils.QuantizationMode.QLinearOps, static: bool, weight_qType: Any, activation_qType: Any, tensors_range: Any, nodes_to_quantize: List[str], nodes_to_exclude: List[str], op_types_to_quantize: List[str], calibrate_method: Any, quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Optional[Dict[str, Any]] = None)




   VitisQDQCPUQuantizer is specific for CPU quantization config.
   Class VitisQDQCPUQuantizer inherits from Class VitisQDQQuantizer and
   can handle float onnx models with inf/-inf initialization.


