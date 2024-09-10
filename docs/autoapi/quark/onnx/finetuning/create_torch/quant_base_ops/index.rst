:orphan:

:py:mod:`quark.onnx.finetuning.create_torch.quant_base_ops`
===========================================================

.. py:module:: quark.onnx.finetuning.create_torch.quant_base_ops


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.create_torch.quant_base_ops.Quantizer
   quark.onnx.finetuning.create_torch.quant_base_ops.AdaroundConstants
   quark.onnx.finetuning.create_torch.quant_base_ops.AdaroundQuantizer
   quark.onnx.finetuning.create_torch.quant_base_ops.QuantizationModule
   quark.onnx.finetuning.create_torch.quant_base_ops.QuantizeWrapper




.. py:class:: Quantizer(scale: torch.Tensor, zero_point: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor, ch_axis: int = 0, q_folded: bool = False)




   Standard Quantizer has three functions including quantize,
   dequantize and quantize_dequantize, which is corresponding to ONNX
   QuantizeLinear, DequantizeLinear and Q/DQ pair separately.
   By default in forward, it works in quantize_dequantize mode.

   .. py:method:: round_impl(tensor: torch.Tensor) -> None

      Implement the round function, designed for adaround quantizer 


   .. py:method:: tensor_sync(tensor: torch.Tensor) -> None

      The Pre-processing of the parameter according to the input tensor 



.. py:class:: AdaroundConstants


   Constants used for Adarounding 


.. py:class:: AdaroundQuantizer(scale: torch.Tensor, zero_point: torch.Tensor, min_q: torch.Tensor, max_q: torch.Tensor, ch_axis: int = 0, q_folded: bool = False)




   Adaround Quantizer has a alpha paramter for optimizing weight rounding 

   .. py:method:: round_impl(tensor: torch.Tensor) -> None

      Implement the rounding function for adaround
      :param weight: The tensor to be ada-rounded


   .. py:method:: initialize_alpha(tensor: torch.Tensor) -> None

      Initializes alpha parameter, same shape as the tensor
      :param tensor: The tensor to be ada-rounded



.. py:class:: QuantizationModule(quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool], Dict[str, Any], None])




   A pytorch module that behaves as ONNX quantization nodes 


.. py:class:: QuantizeWrapper(w_alpha: float = 1.0, b_beta: float = 1.0, **kwargs: Dict[str, Any])




   A wrapper for torch layer's input/weight/bias quantization 


