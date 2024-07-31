Quark for ONNX - Tools
======================

-  `Convert a float16 model to a float32
   model <#convert-a-float16-model-to-a-float32-model>`__
-  `Convert a NCHW input model to a NHWC
   model <#convert-a-nchw-input-model-to-a-nhwc-model>`__
-  `Quantize ONNX model using random
   input <#quantize-onnx-model-using-random-input>`__
-  `Convert a A8W8 NPU model to a A8W8 CPU
   model <#convert-a-a8w8-npu-model-to-a-a8w8-cpu-model>`__

Convert a float16 model to a float32 model
------------------------------------------

Since the Quark ONNX tool only supports float32 models quantization
currently, converting a model from float16 to float32 is required when
quantizing a float16 model.

Use the convert_fp16_to_fp32 tool to convert a float16 model to a
float32 model:

::

   python -m pip install onnxsim
   python -m quark.onnx.tools.convert_fp16_to_fp32 --input $FLOAT_16_ONNX_MODEL_PATH --output $FLOAT_32_ONNX_MODEL_PATH

**Note**: When using convert_fp16_to_fp32 in Quark ONNX, it requires
onnxsim to simplify the ONNX model. Please make sure that onnxsim is
installed by using ‘python -m pip install onnxsim’.

Convert a NCHW input model to a NHWC model
------------------------------------------

Given that some models are designed with an input shape of NCHW instead
of NHWC, it’s recommended to convert an NCHW input model to NHWC before
quantizing a float32 model.

Use the convert_nchw_to_nhwc tool to convert a NCHW model to a NHWC
model:

::

   python -m quark.onnx.tools.convert_nchw_to_nhwc --input $NCHW_ONNX_MODEL_PATH --output $NHWC_ONNX_MODEL_PATH

Quantize ONNX model using random input
--------------------------------------

Given some ONNX models without input for quantization, use random input
for the onnx model quantization process.

Use the random_quantize tool to quantize a onnx model:

::

   python -m quark.onnx.tools.random_quantize --input_model $FLOAT_ONNX_MODEL_PATH --quant_model $QUANTIZED_ONNX_MODEL_PATH

Convert a A8W8 NPU model to a A8W8 CPU model
--------------------------------------------

Given that some models are quantized by A8W8 NPU, it’s convenient and
efficient to convert them to A8W8 CPU models.

Use the convert_a8w8_npu_to_a8w8_cpu tool to convert a A8W8 NPU model to
a A8W8 CPU model:

::

   python -m quark.onnx.tools.convert_a8w8_npu_to_a8w8_cpu --input [INPUT_PATH] --output [OUTPUT_PATH]

Print names and quantity of A16W8 and A8W8 Conv for mix-precision models
------------------------------------------------------------------------

Given that some models are mixed precision such as A18W8 and A8W8 mixed.

Use the print_a16w8_a8w8_nodes tool to print names and quantity of A16W8
and A8W8 Conv, ConvTranspose and Gemm:

::

   python -m quark.onnx.tools.print_a16w8_a8w8_nodes --input [INPUT_PATH]

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
