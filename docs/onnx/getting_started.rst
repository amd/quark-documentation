Getting Started with Quark for ONNX
===================================

Here is an introductory example of running a quantization.

.. code-block:: python

   from onnxruntime.quantization.calibrate import CalibrationDataReader
   from quark.onnx.quantization.config import Config, get_default_config
   from quark.onnx import ModelQuantizer

    # Define model paths
    # Path to the float model to be quantized
    float_model_path = "path/to/float_model.onnx"
    # Path where the quantized model will be saved
    quantized_model_path = "path/to/quantized_model.onnx"
    calib_data_folder = "path/to/calibration_data"
    model_input_name = 'model_input_name'

    # Define calibration data reader for static quantization
    class CalibDataReader(CalibrationDataReader):
        def __init__(self, calib_data_folder: str, model_input_name: str):
            self.input_name = model_input_name
            self.data = self._load_calibration_data(calib_data_folder)
            self.data_iter = None

        # Customize this function to preprocess calibration datasets as needed
        def _load_calibration_data(self, data_folder: str):
            # Example: Implement the actual data preprocessing here
            processed_data = []
            """
            Define preprocessing steps for your dataset.
            For instance, read images and apply necessary transformations.
            """
            return processed_data

        def get_next(self):
            if self.data_iter is None:
                self.data_iter = iter([{self.input_name: data} for data in self.data])
            return next(self.data_iter, None)

    # Instantiate the calibration data reader
    calib_data_reader = CalibDataReader(calib_data_folder, model_input_name)

    # Set up quantization with a specified configuration
    # For example, use "XINT8" for Ryzen AI INT8 quantization
    xint8_config = get_default_config("XINT8")
    quantization_config = Config(global_quant_config=xint8_config )
    quantizer = ModelQuantizer(quantization_config)

    # Quantize the ONNX model and save to specified path
    quantizer.quantize_model(float_model_path, quantized_model_path, calib_data_reader)

To define a calibration dataset for static quantization, refer to :doc:`Adding Calibration Datasets<./user_guide_datareader>`.

For more details about the quantization configuration, refer to :doc:`Configuring Quark for ONNX<./user_guide_config_description>`.

For complete and detailed quantization examples, refer to :doc:`Examples<./onnx_examples>`.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
