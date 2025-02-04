:py:mod:`quark.onnx.tools.convert_opset_version`
================================================

.. py:module:: quark.onnx.tools.convert_opset_version

.. autoapi-nested-parse::

   Convert the opset version of input model.

   :param input: the path of input model
   :param target_opset: the target opset version
   :param output: the path of output model

   Use the convert_opset_version to convert a model's opset version:

   ```
   python convert_opset_version.py --input $INPUT_ONNX_MODEL_PATH --target_opset &TARGET_OPSET_VERSION --output $OUTPUT_ONNX_MODEL_PATH
   ```



