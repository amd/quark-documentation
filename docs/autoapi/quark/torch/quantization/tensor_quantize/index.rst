:orphan:

:py:mod:`quark.torch.quantization.tensor_quantize`
==================================================

.. py:module:: quark.torch.quantization.tensor_quantize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.tensor_quantize.FakeQuantizeBase
   quark.torch.quantization.tensor_quantize.FakeQuantize




.. py:class:: FakeQuantizeBase(device: Optional[torch.device] = None)




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.


   .. py:method:: update_buffer(buffer_name: str, new_value: Union[torch.Tensor, None], input_tensor_device: torch.device) -> None

      Update the value of a registered buffer while ensuring that its shape,
      device, and data type match the input tensor.

      Parameters:
      - buffer_name: The name of the buffer to update
      - new_value: The new value to assign to the buffer
      - input_tensor_device: The target device (e.g., torch.device('cuda') or torch.device('cpu'))



.. py:class:: FakeQuantize(quant_spec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, **kwargs: Any)




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.



