:py:mod:`quark.onnx.quantization.config.config`
===============================================

.. py:module:: quark.onnx.quantization.config.config

.. autoapi-nested-parse::

   Quark Quantization Config API for ONNX



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.quantization.config.config.Config
   quark.onnx.quantization.config.config.QuantizationConfig




.. py:class:: Config


   A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

   :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.


.. py:class:: QuantizationConfig


   A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

   .. attribute:: calibrate_method

      Method used for calibration. Default is CalibrationMethod.MinMax.

      :type: Union[CalibrationMethod, PowerOfTwoMethod]

   .. attribute:: quant_format

      Format of quantization. Default is QuantFormat.QDQ.

      :type: Union[QuantFormat, VitisQuantFormat]

   .. attribute:: activation_type

      Type of quantization for activations. Default is QuantType.QInt8.

      :type: Union[QuantType, VitisQuantType]

   .. attribute:: weight_type

      Type of quantization for weights. Default is QuantType.QInt8.

      :type: Union[QuantFormat, VitisQuantFormat]

   .. attribute:: input_nodes

      List of input nodes to be quantized. Default is an empty list.

      :type: List[str]

   .. attribute:: output_nodes

      List of output nodes to be quantized. Default is an empty list.

      :type: List[str]

   .. attribute:: op_types_to_quantize

      List of operation types to be quantized. Default is an empty list.

      :type: List[str]

   .. attribute:: nodes_to_quantize

      List of node names to be quantized. Default is an empty list.

      :type: List[str]

   .. attribute:: nodes_to_exclude

      List of node names to be excluded from quantization. Default is an empty list.

      :type: List[str]

   .. attribute:: specific_tensor_precision

      Flag to enable specific tensor precision. Default is False.

      :type: bool

   .. attribute:: execution_providers

      List of execution providers. Default is ['CPUExecutionProvider'].

      :type: List[str]

   .. attribute:: per_channel

      Flag to enable per-channel quantization. Default is False.

      :type: bool

   .. attribute:: reduce_range

      Flag to reduce quantization range. Default is False.

      :type: bool

   .. attribute:: optimize_model

      Flag to optimize the model. Default is False.

      :type: bool

   .. attribute:: use_external_data_format

      Flag to use external data format. Default is False.

      :type: bool

   .. attribute:: convert_fp16_to_fp32

      Flag to convert FP16 to FP32. Default is False.

      :type: bool

   .. attribute:: convert_nchw_to_nhwc

      Flag to convert NCHW to NHWC. Default is False.

      :type: bool

   .. attribute:: include_sq

      Flag to include square root in quantization. Default is False.

      :type: bool

   .. attribute:: include_cle

      Flag to include CLE in quantization. Default is False.

      :type: bool

   .. attribute:: include_auto_mp

      Flag to include automatic mixed precision. Default is False.

      :type: bool

   .. attribute:: include_fast_ft

      Flag to include fast fine-tuning. Default is False.

      :type: bool

   .. attribute:: enable_npu_cnn

      Flag to enable NPU CNN. Default is False.

      :type: bool

   .. attribute:: enable_npu_transformer

      Flag to enable NPU Transformer. Default is False.

      :type: bool

   .. attribute:: debug_mode

      Flag to enable debug mode. Default is False.

      :type: bool

   .. attribute:: print_summary

      Flag to print summary of quantization. Default is True.

      :type: bool

   .. attribute:: extra_options

      Dictionary for additional options. Default is an empty dictionary.

      :type: Dict[str, Any]


