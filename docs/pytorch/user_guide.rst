Quark for PyTorch
==========

There are several steps to quantize a floating-point model with
``Quark for PyTorch``:

1. Load original float model
2. Set quantization configuration
3. Define dataloader
4. Use the Quark API to perform in-place replacement of the model's modules with quantized module.
5. (Optional) Export quantized model to other format such as ONNX

More details:
   
* `Configuring Quark for PyTorch <./pytorch/user_guide_config_description.html>`__
* `Adding Calibration Datasets <./pytorch/user_guide_dataloader.html>`__
* `Exporting for ONNX & Json-Safetensors & GGUF <./pytorch/user_guide_exporting.html>`__
* `Feature Description <./pytorch/user_guide_feature_description.html>`__


.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
