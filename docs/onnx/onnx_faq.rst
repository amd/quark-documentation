Frequently Asked Questions (FAQ)
================================


Quark for ONNX
--------------

Model Issues
~~~~~~~~~~~~

**Issue 1**:

Error of "ValueError:Message onnx.ModelProto exceeds maximum protobuf size of 2GB"

**Solution**:

This error is caused by the input model size exceeding 2GB. Please set optimize_model=False and use_external_data_format=True.

Quantization Issues
~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Error of "onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node."

**Solution**:

For networks with an ROI head, such as Mask R-CNN or Faster R-CNN, quantization errors may arise if ROIs are not generated in the network.
Please use quark.onnx.PowerOfTwoMethod.MinMSE or quark.onnx.CalibrationMethod.Percentile quantization and perform inference with real data.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
