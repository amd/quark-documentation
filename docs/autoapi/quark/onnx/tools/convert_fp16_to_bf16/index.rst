:py:mod:`quark.onnx.tools.convert_fp16_to_bf16`
===============================================

.. py:module:: quark.onnx.tools.convert_fp16_to_bf16

.. autoapi-nested-parse::

   Convert tensor float16 type in the ONNX ModelProto input to tensor bfloat16.

   Use the convert_fp16_to_bf16.py to convert a float16 model to a bfloat16 model:

   ```
   python convert_fp16_to_bf16.py --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH
   ```



