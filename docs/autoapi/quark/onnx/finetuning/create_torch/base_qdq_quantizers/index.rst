:orphan:

:py:mod:`quark.onnx.finetuning.create_torch.base_qdq_quantizers`
================================================================

.. py:module:: quark.onnx.finetuning.create_torch.base_qdq_quantizers


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.create_torch.base_qdq_quantizers.INTQuantizer
   quark.onnx.finetuning.create_torch.base_qdq_quantizers.AdaroundConstants
   quark.onnx.finetuning.create_torch.base_qdq_quantizers.AdaroundINTQuantizer
   quark.onnx.finetuning.create_torch.base_qdq_quantizers.FPQuantizer




.. py:class:: INTQuantizer(scale: torch.Tensor, zero_point: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor, ch_axis: int = 0, q_folded: bool = False)




   Standard integer quantizer has three functions including quantize,
   dequantize and quantize_dequantize, which is corresponding to ONNX
   QuantizeLinear, DequantizeLinear and Q/DQ pair separately.
   By default in forward, it works in quantize_dequantize mode.

   .. py:method:: round_impl(tensor: torch.Tensor) -> None

      Implement the round function, designed for adaround quantizer 


   .. py:method:: tensor_sync(tensor: torch.Tensor) -> None

      The Pre-processing of the parameter according to the input tensor 



.. py:class:: AdaroundConstants


   Constants used for Adarounding 


.. py:class:: AdaroundINTQuantizer(scale: torch.Tensor, zero_point: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor, ch_axis: int = 0, q_folded: bool = False)




   AdaRound integer quantizer has a alpha paramter for optimizing weight rounding 

   .. py:method:: round_impl(tensor: torch.Tensor) -> None

      Implement the rounding function for adaround
      :param weight: The tensor to be ada-rounded


   .. py:method:: initialize_alpha(tensor: torch.Tensor) -> None

      Initializes alpha parameter, same shape as the tensor
      :param tensor: The tensor to be ada-rounded



.. py:class:: FPQuantizer(scale: torch.Tensor, zero_point: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor, ch_axis: int = 0, q_folded: bool = False, quant_type: torch.dtype = torch.bfloat16)




   Standard floating point quantizer, such as quantizer for Float16 and BFloat16 quantization.
   There are still scale and zp for the quantization to do the scaling and shift.


