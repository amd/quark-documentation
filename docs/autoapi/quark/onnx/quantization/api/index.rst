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

   :param config: Configuration object containing settings for quantization.
   :type config: Config

   .. note::

      It is essential to ensure that the 'config' provided has all necessary quantization parameters defined.
      This class assumes that the model is compatible with the quantization settings specified in 'config'.

   .. py:method:: quantize_model(model_input: str, model_output: str, calibration_data_reader: Union[onnxruntime.quantization.calibrate.CalibrationDataReader, None] = None) -> None

      Quantizes the given ONNX model and saves the output to the specified path.

      :param model_input: Path to the input ONNX model file.
      :type model_input: str
      :param model_output: Path where the quantized ONNX model will be saved.
      :type model_output: str
      :param calibration_data_reader: Data reader for model calibration. Defaults to None.
      :type calibration_data_reader: Union[CalibrationDataReader, None], optional

      :returns: None

      :raises ValueError: If the input model path is invalid or the file does not exist.



