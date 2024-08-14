Frequently Asked Questions (FAQ)
================================

How to give feedback
--------------------

If AMD internal users need to report an issue or make a request, they should file a JIRA ticket under the project named "quark" in the
`internal JIRA
system <https://jira.xilinx.com/secure/CreateIssue!default.jspa>`__.

Quark for Pytorch
-----------------

Environment Issues
~~~~~~~~~~~~~~~~~~

**Issue 1**:

Windows CPU mode does not support fp16.

**Solution**:

Because of torch `issue <https://github.com/pytorch/pytorch/issues/52291>`__\ , Windows CPU mode cannot perfectly support fp16.

C++ Compilation Issues
~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Stuck in the compilation phase for a long time (over ten minutes), the terminal shows like:

.. code:: shell

   [QUARK-INFO]: Configuration checking start. 

   [QUARK-INFO]: C++ kernel build directory [cache folder path]/torch_extensions/py39...

**Solution**:

delete the cache folder ``[cache folder path]/torch_extensions`` and run quark again.

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
