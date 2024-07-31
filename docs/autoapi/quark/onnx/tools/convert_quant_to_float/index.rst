:py:mod:`quark.onnx.tools.convert_quant_to_float`
=================================================

.. py:module:: quark.onnx.tools.convert_quant_to_float

.. autoapi-nested-parse::

   Convert quantized model to FP32 model.



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.convert_quant_to_float.convert_initializers_to_float



.. py:function:: convert_initializers_to_float(model: onnx.ModelProto, initializers_to_convert: Dict[str, Dict[str, str]]) -> onnx.ModelProto

   Convert integer initializers used by DequantizeLinear nodes to float initializers.


