:orphan:

:py:mod:`quark.onnx.finetuning.auto_mixprecision`
=================================================

.. py:module:: quark.onnx.finetuning.auto_mixprecision


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.auto_mixprecision.auto_mixprecision



.. py:function:: auto_mixprecision(float_model_path: str, quant_model_path: str, dr: Any, activation_type: Any, weight_type: Any, extra_options: Any) -> Any

   Automatic apply low precision quantization on Q/DQ.


