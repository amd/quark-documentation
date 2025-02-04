:py:mod:`quark.onnx.tools.convert_fp32_to_fp16`
===============================================

.. py:module:: quark.onnx.tools.convert_fp32_to_fp16

.. autoapi-nested-parse::

   Convert tensor float type in the ONNX ModelProto input to tensor float16.

   :param model: ONNX ModelProto object
   :param disable_shape_infer: Type/shape information is needed for conversion to work.
                               Set to True only if the model already has type/shape information for all tensors.
   :return: converted ONNX ModelProto object

   Examples:

   ::

       Example 1: Convert ONNX ModelProto object:
       import float16
       new_onnx_model = float16.convert_float_to_float16(onnx_model)

       Example 2: Convert ONNX model binary file:
       import onnx
       import float16
       onnx_model = onnx.load_model('model.onnx')
       new_onnx_model = float16.convert_float_to_float16(onnx_model)
       onnx.save_model(new_onnx_model, 'new_model.onnx')

   Use the convert_float32_to_float16.py to convert a float32 model to a float16 model:

   ```
   python convert_fp32_to_fp16.py --input $FLOAT_32_ONNX_MODEL_PATH --output $FLOAT_16_ONNX_MODEL_PATH
   ```

   The conversion from float32 models to float16 models may result in
   the generation of unnecessary operations such as casts in the model.
   It is recommended to use onnx-simplifier to remove these redundant nodes.



