:py:mod:`quark.onnx.tools.convert_fp16_to_bfp16`
================================================

.. py:module:: quark.onnx.tools.convert_fp16_to_bfp16

.. autoapi-nested-parse::

   Convert tensor float16 type in the ONNX ModelProto input to tensor bfp16.

   Use the convert_fp16_to_bfp16.py to convert a float16 model to a bfp16 model:

   ```
   python convert_fp16_to_bfp16.py --input $FLOAT_16_ONNX_MODEL_PATH --output $BFP_16_ONNX_MODEL_PATH
   ```



