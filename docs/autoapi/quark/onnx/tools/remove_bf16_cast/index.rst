:py:mod:`quark.onnx.tools.remove_bf16_cast`
===========================================

.. py:module:: quark.onnx.tools.remove_bf16_cast

.. autoapi-nested-parse::

   Rmove bfloat16 cast ops for an onnx model.

   :param input_model_path: the path of input bfloat16 quantized model with bfloat16 cast
   :param output_model_path: the path of bfloat16 quantized model with no bfloat16 cast

   Use the remove_bf16_cast.py to remove bfloat16 cast for a bfloat16 quantized model:

   ```
   python remove_bf16_cast.py --input $INPUT_MODEL_PATH --output $OUTPUT_MODEL_PATH
   ```



