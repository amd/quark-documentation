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
   
* `Configuring Quark for PyTorch <./pytorch/user_guide_config_description.rst>`__
* `Adding Calibration Datasets <./pytorch/user_guide_dataloader.rst>`__
* `Exporting for ONNX & Json-Safetensors & GGUF <./pytorch/user_guide_exporting.rst>`__
* `Feature Description <./pytorch/user_guide_feature_description.rst>`__

Quark for ONNX
--------------

There are several steps to quantize a floating-point model with
``Quark for ONNX``:

1. Load original float model
2. Set quantization configuration
3. Define datareader
4. Use the Quark API to perform in-place replacement of the model's modules with quantized module.

More details:

* `Configuring Quark for ONNX <./onnx/user_guide_config_description.rst>`__
* `Adding Calibration Datasets <./onnx/user_guide_datareader.rst>`__
* `Feature Description <./onnx/user_guide_feature_description.rst>`__
* `Supported Datatype and OpType <./onnx/user_guide_supported_optype_datatype.rst>`__
* `Accuracy Improvement <./onnx/user_guide_accuracy_improvement.rst>`__
* `Optional Utilities <./onnx/user_guide_optional_utilities.rst>`__
* `Tools <./onnx/user_guide_tools.rst>`__

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
