User Guide
==========

Quark for ONNX
--------------

There are several steps to quantize a floating-point model with
``Quark for ONNX``:

1. Load original float model
2. Set quantization configuration
3. Define datareader
4. Use the Quark API to perform in-place replacement of the model's modules with quantized module.

More details:

* :doc:`Configuring Quark for ONNX <user_guide_config_description>`
* :doc:`Adding Calibration Datasets <user_guide_datareader>`
* :doc:`Feature Description <user_guide_feature_description>`
* :doc:`Supported Data type and Op Type <user_guide_supported_optype_datatype>`
* :doc:`Accuracy Improvement <user_guide_accuracy_improvement>`
* :doc:`Optional Utilities <user_guide_optional_utilities>`
* :doc:`GPU Usage Guide <gpu_usage_guide>`
* :doc:`Tools <user_guide_tools>`

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
