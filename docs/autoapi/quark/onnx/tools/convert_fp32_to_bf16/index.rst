:py:mod:`quark.onnx.tools.convert_fp32_to_bf16`
===============================================

.. py:module:: quark.onnx.tools.convert_fp32_to_bf16

.. autoapi-nested-parse::

   Convert tensor float type in the ONNX ModelProto input to tensor bfloat16.

   Use the convert_fp32_to_bf16.py to convert a float32 model to a bfloat16 model:

   ```
   python convert_fp32_to_bf16.py --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH
   ```



