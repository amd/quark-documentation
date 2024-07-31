:orphan:

:py:mod:`quark.torch.algorithm.rotation.hadamard`
=================================================

.. py:module:: quark.torch.algorithm.rotation.hadamard


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.rotation.hadamard.random_hadamard_matrix
   quark.torch.algorithm.rotation.hadamard.get_hadamard_matrices
   quark.torch.algorithm.rotation.hadamard.hardmard_transform



.. py:function:: random_hadamard_matrix(size: int) -> torch.Tensor

   Generate a random Hadamard matrix of size `size`.


.. py:function:: get_hadamard_matrices(n: int) -> tuple[torch.Tensor, Optional[torch.Tensor], int]

   Get the Hadamard matrix and its dimension for a given input size.


.. py:function:: hardmard_transform(X: torch.Tensor, H1: torch.Tensor, H2: Optional[torch.Tensor], K: int, scaled: bool = False) -> torch.Tensor

   Apply Hadamard matrix to the input tensor.


