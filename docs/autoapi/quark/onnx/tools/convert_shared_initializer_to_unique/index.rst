:py:mod:`quark.onnx.tools.convert_shared_initializer_to_unique`
===============================================================

.. py:module:: quark.onnx.tools.convert_shared_initializer_to_unique

.. autoapi-nested-parse::

   Convert ONNX ModelProto with shared initializer to be unique initializer.

   :param model: ONNX ModelProto object
   :param op_types: The op_type need to copy the shared initializer
   :return: converted ONNX ModelProto object

   Examples:

   ::

       Example 1: Convert ONNX ModelProto object:
       from quark.onnx.tools import convert_shared_initializer_to_unique
       new_onnx_model = convert_shared_initializer_to_unique.convert(onnx_model)


   Use the convert_shared_initializer_to_unique.py to duplicate reused initializer in onnx model,
   so that there do not exist nodes to share initializer

   ```
   python convert_shared_initializer_to_unique.py --input $ONNX_MODEL_PATH_WITH_INIT_SHARED --output $ONNX_MODEL_PATH_WITHOUT_INIT_SHARED --op_types ["Cnv", "Gemm"]
   ```

   If need to duplicate all op_types in the given onnx model, the op_types could include all op_types or keep None.

   ```
   python convert_shared_initializer_to_unique.py --input $ONNX_MODEL_PATH_WITH_INIT_SHARED --output $ONNX_MODEL_PATH_WITHOUT_INIT_SHARED
   ```


   The conversion from reused initializer to that one without initializer shared
   for given node op_types e.g. ["Conv", "Gemm"]. Empty list [] will include all
   op_types in the given onnx model, default is [].
   It is recommended to do conversion to satisfy the compilation need and model
   quantization FastFinetune need.



