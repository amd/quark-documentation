Quark for ONNX - Adding Calibration Datasets
============================================

Class DataReader to Quark quantizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark for ONNX utilizes [ONNXRuntime Quantization Dataloader] for
normalization during quantization calibration. The code below is an
example showing how to define the class of calibration data loader.

.. code:: python

   import onnxruntime
   from onnxruntime.quantization.calibrate import CalibrationDataReader

   class ImageDataReader(CalibrationDataReader):

       def __init__(self, calibration_image_folder: str, input_name: str,
        input_height: int, input_width: int):
           self.enum_data = None

           self.input_name = input_name

           self.data_list = self._preprocess_images(
                   calibration_image_folder, input_height, input_width)

       # The pre-processing of calibration images should be defined by users.
       # Recommended batch_size is 1. 
       def _preprocess_images(self, image_folder: str, input_height: int, input_width: int, batch_size: int = 1):
           data_list = []
           '''
           The pre-processing for each image
           '''
           return data_list

       def get_next(self):
           if self.enum_data is None:
               self.enum_data = iter([{self.input_name: data} for data in self.data_list])
           return next(self.enum_data, None)

       def rewind(self):
           self.enum_data = None

   input_model_path = "path/to/your/resnet50.onnx"
   output_model_path = "path/to/your/resnet50_quantized.onnx"
   calibration_image_folder = "path/to/your/images"

   input_name = 'input_tensor_name'
   input_shape = (1, 3, 224, 224)
   calib_datareader = ImageDataReader(calibration_image_folder, input_name,
    input_shape[2], input_shape[3])

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
