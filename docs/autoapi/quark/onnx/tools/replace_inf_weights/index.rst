:py:mod:`quark.onnx.tools.replace_inf_weights`
==============================================

.. py:module:: quark.onnx.tools.replace_inf_weights

.. autoapi-nested-parse::

   A tool for replace `inf` and `-inf` values in ONNX model weights with specified replacement values.'

       Example : python -m quark.onnx.tools.replace_inf_weights --input_model [INPUT_MODEL_PATH] --output_model [OUTPUT_MODEL_PATH] --replace_inf_value [REPLACE_INF_VALUE]



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.replace_inf_weights.replace_inf_in_onnx_weights



.. py:function:: replace_inf_in_onnx_weights(input_model_path: str, output_model_path: str, replace_inf_value: float = 10000.0) -> None

   Replaces `inf` and `-inf` values in the weights of an ONNX model with specified default values.

   Parameters:
       input_model_path (str): Path to the input ONNX model file.
       output_model_path (str): Path to save the modified ONNX model file.
       replace_inf_value (float): The base value used to replace `inf` and `-inf`.
                             - Positive `inf` values are replaced with `replace_inf_value`.
                             - Negative `inf` values are replaced with `-replace_inf_value`.

   Returns:
       None: The function directly modifies the model and saves it to the output path.


