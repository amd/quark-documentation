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
   quark.torch.quantization.tensor_quantize.ScaledFakeQuantize
   quark.torch.quantization.tensor_quantize.FreezedScaledFakeQuantize
   quark.torch.quantization.tensor_quantize.NonScaledFakeQuantize




.. py:class:: FakeQuantizeBase(quant_spec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




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



.. py:class:: ScaledFakeQuantize(quant_spec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, **kwargs: Any)




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: FreezedScaledFakeQuantize(dtype: quark.torch.quantization.config.type.Dtype, quant_spec: quark.torch.quantization.config.config.QuantizationSpec)




   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


.. py:class:: NonScaledFakeQuantize(quant_spec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.



