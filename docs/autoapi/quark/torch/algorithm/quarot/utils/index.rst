:orphan:

:py:mod:`quark.torch.algorithm.quarot.utils`
============================================

.. py:module:: quark.torch.algorithm.quarot.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.quarot.utils.QKRotation
   quark.torch.algorithm.quarot.utils.R4Wrapper



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.quarot.utils.rotate_in_channels2
   quark.torch.algorithm.quarot.utils.rotate_out_channels2
   quark.torch.algorithm.quarot.utils.hadamard_transform
   quark.torch.algorithm.quarot.utils.hadamard_multiply
   quark.torch.algorithm.quarot.utils.add_qk_rotation_after_function_call_in_forward



.. py:function:: rotate_in_channels2(module: torch.nn.Module, rotation: torch.Tensor) -> None

   Rotate the input channels of a linear layer.
   If weight and rotation's sizes don't match, it reshapes weight in order to multiply them.


.. py:function:: rotate_out_channels2(module: torch.nn.Module, rotation: torch.Tensor) -> None

   Rotate the output channels of a linear layer.
   If weight/bias and rotation's sizes don't match
   it reshapes weight/bias in order to multiply them.


.. py:function:: hadamard_transform(x: torch.Tensor) -> torch.Tensor

   Applies Hadamard transform to x (without dividing by sqrt n). Ideally should be replaced by a hardware
   optimized kernel, since Hadamard transforms can in theory be done much faster than general matrix multiplications.

   Code from: https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py


.. py:function:: hadamard_multiply(x: torch.Tensor) -> torch.Tensor

   Applies hadamard transform to x with dividing by sqrt n 


.. py:class:: QKRotation(func: Callable[Ellipsis, Any])




   Performs R3 rotation after RoPE of both Q and K, but does not do K quantization


.. py:function:: add_qk_rotation_after_function_call_in_forward(module: torch.nn.Module, function_name: str) -> None

   This function adds a rotation wrapper after the output of a function call in forward.
   Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.

   This function used to insert the R3 rotation after the output of the call of the RoPE operation.
   Implementating it like this is not ideal, since we need to modify the forward function's globals. However, this is the
   trick used by both QuaRot and SpinQuant to insert a rotation after the RoPE operation. Ultimately it would better to
   find a way to implement this feature without touching globals.


.. py:class:: R4Wrapper(module: torch.nn.Module)




   Wrapper around a nn.Module that applies a Hadamard rotation before the module.
   If the module is an nn.Linear or nn.Conv, then Quark will replace it by a quantized linear layer
   If there is activation quantization, it is applied in between, i.e. after the rotation
   but before the forward pass of the module


