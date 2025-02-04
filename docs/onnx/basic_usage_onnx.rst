AMD Quark for ONNX
==================

Quantizing a floating-point model using AMD Quark for ONNX involves several key steps:

1. **Load the Original Float Model**

   Load your model in its original floating-point format.

2. **Set Quantization Configuration**

   Specify the quantization settings, such as precision and calibration parameters.

3. **Define Data Reader**

   Create a data reader to provide input data for model calibration.

4. **Use AMD Quark API for In-Place Replacement**

   Apply the AMD Quark API to replace the model's modules with quantized versions in-place.

Supported Features
-------------------

AMD Quark for ONNX supports the following features:

.. list-table::
   :header-rows: 1

   * - Feature Name
     - Feature Value
   * - Activation/Weight Type
     - Int8 / Uint8 / Int16 / Uint16 / Int32 / Uint32 / Float16 / Bfloat16
   * - Quant Strategy
     - Static quant / Weight only / Dynamic quant
   * - Quant Scheme
     - Per tensor / Per channel
   * - Quant Format
     - QuantFormatQDQ / VitisQuantFormat.QDQ / QuantFormat.QOperator
   * - Calibration Method
     - MinMax / Percentile / MinMSE / Entropy / NonOverflow
   * - Symmetric
     - Symmetric / Asymmetric
   * - Scale Type
     - Float32 / Float16
   * - Pre-Quant Optimization
     - QuaRot / SmoothQuant (Single_GPU/CPU) / CLE / Bias Correction
   * - Quant Algorithm
     - AdaQuant / AdaRound / GPTQ
   * - Operating Systems
     - Linux(ROCm/CUDA) / Windows(CPU)

Basic Example
--------------

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

For more detailed information, see :ref:`Advanced AMD Quark Features for ONNX <advanced-quark-features-onnx>`.



