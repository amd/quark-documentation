Getting Started with Quark for ONNX
===================================

Here is an example of running quantization with
``U8S8_AAWS_CONFIG`` configurations. We also support
quantization without real calibration data for rapid validation of
deployment or performance benchmarking. Detailed explanations for each
step will be provided on other chapter of the User Guide.

.. code:: python

   import onnxruntime
   from onnxruntime.quantization.calibrate import CalibrationDataReader
   from quark.onnx.quantization.config import (Config, get_default_config)
   from quark.onnx import ModelQuantizer

   # 1. Set Model
   # The input_model_path is the path to the floating point model to be quantized. The output_model_path is the path where the quantized model will be saved.
   input_model_path = "/path/to/input/model"
   output_model_path = "/path/to/output/model"

   # 2. Set Calibration Dataset
   # `dr` (Data Reader) is an instance of CalibrationDataReader. When dr is None, the quantizer will use random data for calibration. Please refer to user guide for how to set up the CalibrationDataReader.  
   dr = None

   # 3. Set quantization configuration
   quant_config = get_default_config("U8S8_AAWS")
   config = Config(global_quant_config=quant_config)
   quantizer = ModelQuantizer(config)

   # 5. Quantize the ONNX model
   quantizer.quantize_model(input_model_path, output_model_path, dr)

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
