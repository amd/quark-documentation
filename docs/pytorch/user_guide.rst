User Guide
==========

Quark for PyTorch
-----------------

There are several steps to quantize a floating-point model with
``Quark for PyTorch``:

1. Load original float model
2. Set quantization configuration
3. Define dataloader
4. Use the Quark API to perform in-place replacement of the model's modules with quantized module.
5. (Optional) Export quantized model to other format such as ONNX

More details:
   
* :doc:`Configuring Quark for PyTorch <user_guide_config_description>`
* :doc:`Adding Calibration Datasets <user_guide_dataloader>`
* :doc:`Exporting & Saving & Loading <user_guide_output>`
* :doc:`Feature Description <user_guide_feature_description>`

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
