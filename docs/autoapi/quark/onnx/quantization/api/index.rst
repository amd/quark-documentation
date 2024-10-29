:py:mod:`quark.onnx.quantization.api`
=====================================

.. py:module:: quark.onnx.quantization.api

.. autoapi-nested-parse::

   Quark Quantization API for ONNX.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.quantization.api.ModelQuantizer




.. py:class:: ModelQuantizer(config: quark.onnx.quantization.config.config.Config)


   Provides an API for quantizing deep learning models using ONNX. This class handles the
   configuration and processing of the model for quantization based on user-defined parameters.

   Args:
       config (Config): Configuration object containing settings for quantization.

   Note:
       It is essential to ensure that the 'config' provided has all necessary quantization parameters defined.
       This class assumes that the model is compatible with the quantization settings specified in 'config'.

   .. py:method:: quantize_model(model_input: str, model_output: str, calibration_data_reader: Union[onnxruntime.quantization.calibrate.CalibrationDataReader, None] = None, calibration_data_path: Optional[str] = None) -> None

      Quantizes the given ONNX model and saves the output to the specified path.

      Args:
          model_input (str): Path to the input ONNX model file.
          model_output (str): Path where the quantized ONNX model will be saved.
          calibration_data_reader (Union[CalibrationDataReader, None], optional): Data reader for model calibration. Defaults to None.

      Returns:
          None

      Raises:
          ValueError: If the input model path is invalid or the file does not exist.



