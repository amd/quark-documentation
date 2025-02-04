:orphan:

:py:mod:`quark.onnx.refine`
===========================

.. py:module:: quark.onnx.refine


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.refine.adjust_quantize_info
   quark.onnx.refine.align_quantize_info



.. py:function:: adjust_quantize_info(model: onnx.ModelProto, max_loop_num: int = 5, adjust_shift_cut: bool = True, adjust_shift_bias: bool = True, adjust_shift_read: bool = True, adjust_shift_write: bool = True, adjust_hard_sigmoid: bool = True, adjust_shift_swish: bool = True, align_concat: bool = True, align_pool: bool = True, align_pad: bool = True, align_slice: bool = True) -> quark.onnx.quant_utils.ONNXQuantizedModel

   Adjust the quantize info to meet the compiler constraints.


.. py:function:: align_quantize_info(model: onnx.ModelProto, max_loop_num: int = 5, align_concat: bool = True, align_pool: bool = True, align_pad: bool = True, align_slice: bool = True, align_transpose: bool = True, align_reshape: bool = True, adjust_bias_scale: bool = True) -> Any

   Align the quantize info to meet the compiler constraints.
   This function supports pof2 scale and float scale both


