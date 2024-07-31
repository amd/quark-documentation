:orphan:

:py:mod:`quark.onnx.calibrate`
==============================

.. py:module:: quark.onnx.calibrate


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.calibrate.MinMaxCalibrater
   quark.onnx.calibrate.EntropyCalibrater
   quark.onnx.calibrate.PercentileCalibrater
   quark.onnx.calibrate.PowOfTwoCalibrater
   quark.onnx.calibrate.PowOfTwoCollector



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.calibrate.create_calibrator_power_of_two
   quark.onnx.calibrate.create_calibrator_float_scale



.. py:class:: MinMaxCalibrater(model_path: pathlib.Path, op_types_to_calibrate: Union[List[str], None], augmented_model_path: str = 'augmented_model.onnx', symmetric: bool = False, use_external_data_format: bool = False, moving_average: bool = False, averaging_constant: float = 0.01)




   This method obtains the quantization parameters based on the minimum and maximum values of each tensor.

   :param model_path: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
   :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
   :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
   :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
   :param moving_average: Whether to compute the moving average of the minimum and maximum values instead of the global minimum and maximum. Default is False.
   :param averaging_constant: Constant smoothing factor to use when computing the moving average. Default is 0.01. Should be between 0 and 1.
   :raises ValueError: If averaging_constant is not between 0 and 1 when moving_average is True.


.. py:class:: EntropyCalibrater(model_path: pathlib.Path, op_types_to_calibrate: Union[List[str], None], augmented_model_path: str = 'augmented_model.onnx', use_external_data_format: bool = False, method: str = 'entropy', symmetric: bool = False, num_bins: int = 128, num_quantized_bins: int = 128)




   This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.

   :param model_path: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
   :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
   :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
   :param method: Method for calibration. One of ['entropy', 'percentile', 'distribution']. Default is "entropy".
   :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
   :param num_bins: Number of bins to create a new histogram for collecting tensor values. Default is 128.
   :param num_quantized_bins: Number of quantized bins. Default is 128.


.. py:class:: PercentileCalibrater(model_path: pathlib.Path, op_types_to_calibrate: Union[List[str], None], augmented_model_path: str = 'augmented_model.onnx', use_external_data_format: bool = False, method: str = 'percentile', symmetric: bool = False, num_bins: int = 2048, percentile: float = 99.999)




   This method calculates quantization parameters using percentiles of the tensor values.

   :param model_path: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
   :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
   :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
   :param method: Method for calibration. One of ['entropy', 'percentile', 'distribution']. Default is "percentile".
   :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is False.
   :param num_bins: Number of bins to create a new histogram for collecting tensor values. Default is 2048.
   :param percentile: Percentile value for calibration, a float between [0, 100]. Default is 99.999.


.. py:class:: PowOfTwoCalibrater(model: pathlib.Path, op_types_to_calibrate: Optional[Sequence[str]], augmented_model_path: str = 'augmented_model.onnx', use_external_data_format: bool = False, activation_type: Union[onnxruntime.quantization.quant_utils.QuantType, quark.onnx.quant_utils.VitisQuantType] = QuantType.QInt8, method: quark.onnx.quant_utils.PowerOfTwoMethod = PowerOfTwoMethod.MinMSE, symmetric: bool = True, minmse_mode: str = 'All', percentile: float = 99.999, quantized_tensor_type: Dict[Any, Any] = {})




   This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.

   :param model: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate. By default, calibrates all the float32/float16 tensors.
   :param augmented_model_path: Path to save the augmented model. Default is "augmented_model.onnx".
   :param use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is False.
   :param activation_type: Type of quantization for activations. Default is QuantType.QInt8.
   :param method: Calibration method. Default is PowerOfTwoMethod.MinMSE.
   :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is True.
   :param minmse_mode: Mode for the MinMSE method. Default is "All".
   :param percentile: Percentile value for calibration, a float between 0 and 100. Default is 99.999.
   :param quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is an empty dictionary.

   .. py:method:: augment_graph() -> None

      make all quantization_candidates op type nodes as part of the graph output.
      :return: augmented ONNX model


   .. py:method:: collect_data(data_reader: onnxruntime.quantization.calibrate.CalibrationDataReader) -> None

      abstract method: collect the tensors that will be used for range computation. It can be called multiple times.


   .. py:method:: compute_range() -> Any

      Compute the min-max range of tensor
      :return: dictionary mapping: {tensor name: (min value, max value)}



.. py:class:: PowOfTwoCollector(activation_type: Union[onnxruntime.quantization.quant_utils.QuantType, quark.onnx.quant_utils.VitisQuantType] = QuantType.QInt8, method: quark.onnx.quant_utils.PowerOfTwoMethod = PowerOfTwoMethod.MinMSE, symmetric: bool = True, minmse_mode: str = 'All', percentile: float = 99.999, quantized_tensor_type: Dict[Any, Any] = {})




   Collecting PowOfTwoCollector quantize for each tensor. Support MinMSE method.

   :param activation_type: Type of quantization for activations. Default is QuantType.QInt8.
   :param method: Calibration method. Default is PowerOfTwoMethod.MinMSE.
   :param symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is True.
   :param minmse_mode: Mode for the MinMSE method. Default is "All".
   :param percentile: Percentile value for calibration, a float between 0 and 100. Default is 99.999.
   :param quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is an empty dictionary.


   .. py:method:: collect(name_to_arr: Dict[Any, Any]) -> None

      Generate informative data based on given data.
          name_to_arr : dict
              tensor name to NDArray data


   .. py:method:: compute_collection_result() -> Any

      Get the optimal result among collection data.



.. py:function:: create_calibrator_power_of_two(model: pathlib.Path, op_types_to_calibrate: List[str], augmented_model_path: str = 'augmented_model.onnx', activation_type: Union[quark.onnx.quant_utils.VitisQuantType, onnxruntime.quantization.quant_utils.QuantType] = QuantType.QInt8, method: quark.onnx.quant_utils.PowerOfTwoMethod = PowerOfTwoMethod.NonOverflow, use_external_data_format: bool = False, execution_providers: Union[List[str], None] = ['CPUExecutionProvider'], quantized_tensor_type: Dict[Any, Any] = {}, extra_options: Dict[str, Any] = {}) -> Any

   Create a calibrator for power-of-two quantization.

   :param model: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate.
   :param augmented_model_path: Path to save the augmented ONNX model.
   :param activation_type: Type of quantization for activations.
   :param method: Calibration method to use.
   :param use_external_data_format: Whether to use external data format for large models.
   :param execution_providers: List of execution providers for ONNX Runtime.
   :param quantized_tensor_type: Dictionary specifying the quantized tensor type.
   :param extra_options: Additional options for calibrator configuration.
   :return: Initialized calibrator object.


.. py:function:: create_calibrator_float_scale(model: pathlib.Path, op_types_to_calibrate: Union[List[str], None], augmented_model_path: str = 'augmented_model.onnx', calibrate_method: onnxruntime.quantization.calibrate.CalibrationMethod = CalibrationMethod.MinMax, use_external_data_format: bool = False, execution_providers: Union[List[str], None] = ['CPUExecutionProvider'], extra_options: Dict[str, Any] = {}) -> Any

   Create a calibrator for floating-point scale quantization.

   :param model: Path to the ONNX model to calibrate.
   :param op_types_to_calibrate: List of operator types to calibrate. If None, all float32/float16 tensors are calibrated.
   :param augmented_model_path: Path to save the augmented ONNX model.
   :param calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, or Distribution).
   :param use_external_data_format: Whether to use external data format for large models.
   :param execution_providers: List of execution providers for ONNX Runtime.
   :param extra_options: Additional options for calibrator configuration.
   :return: Initialized calibrator object.


