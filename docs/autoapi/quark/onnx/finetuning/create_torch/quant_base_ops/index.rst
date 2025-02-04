:orphan:

:py:mod:`quark.onnx.finetuning.create_torch.quant_base_ops`
===========================================================

.. py:module:: quark.onnx.finetuning.create_torch.quant_base_ops


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.create_torch.quant_base_ops.QuantizationModule
   quark.onnx.finetuning.create_torch.quant_base_ops.QuantizeWrapper




.. py:class:: QuantizationModule(quant_info: Union[Tuple[numpy.typing.NDArray[numpy.float32], numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], numpy.typing.NDArray[Any], int, bool, onnx.onnx_pb.TensorProto], Dict[str, Any], None])




   A pytorch module that behaves as ONNX quantization nodes 


.. py:class:: QuantizeWrapper(w_alpha: float = 1.0, b_beta: float = 1.0, **kwargs: Dict[str, Any])




   A wrapper for torch layer's input/weight/bias quantization 


