:orphan:

:py:mod:`quark.onnx.finetuning.create_torch.create_model`
=========================================================

.. py:module:: quark.onnx.finetuning.create_torch.create_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.create_torch.create_model.TorchModel




.. py:class:: TorchModel(onnx_model: onnx.ModelProto)




   A torch model converted from a onnx model.

   .. py:method:: forward(inputs: torch.Tensor) -> Any

      Support the models with single input and single output 


   .. py:method:: set_weight(weight: numpy.typing.NDArray[numpy.float32]) -> None

      Set the original float weight for the compute module 


   .. py:method:: get_weight() -> Any

      Get the optimized quantized weight of the compute module 


   .. py:method:: set_bias(bias: numpy.typing.NDArray[numpy.float32]) -> None

      Set the original float bias for the compute module 


   .. py:method:: get_bias() -> Any

      Get the optimized quantized bias of the compute module 



