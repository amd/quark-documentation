:orphan:

:py:mod:`quark.torch.algorithm.rotation.rotation_utils`
=======================================================

.. py:module:: quark.torch.algorithm.rotation.rotation_utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.rotation.rotation_utils.RMSNorm



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.rotation.rotation_utils.rotate_in_channels
   quark.torch.algorithm.rotation.rotation_utils.rotate_out_channels
   quark.torch.algorithm.rotation.rotation_utils.get_rotation_matrix



.. py:class:: RMSNorm(hidden_size: int, eps: float = 1e-06)




   Root Mean Square Layer Normalization (RMSNorm).

   .. py:method:: forward(hidden_states: torch.Tensor) -> torch.Tensor

      Apply RMSNorm normalization to hidden states.



.. py:function:: rotate_in_channels(module: torch.nn.Module, rotation: torch.Tensor) -> None

   Rotate the input channels of a weight matrix.


.. py:function:: rotate_out_channels(module: torch.nn.Module, rotation: torch.Tensor) -> None

   Rotate the output channels of a weight matrix.


.. py:function:: get_rotation_matrix(num_channels: int, random: bool = True) -> torch.Tensor

   Get a random rotation matrix for the given number of channels.


