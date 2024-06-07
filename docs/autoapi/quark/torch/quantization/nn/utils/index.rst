:orphan:

:py:mod:`quark.torch.quantization.nn.utils`
===========================================

.. py:module:: quark.torch.quantization.nn.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.nn.utils.check_min_max_valid



.. py:function:: check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool

   Checks if the given minimum and maximum values are valid, meaning that
   they exist and the min value is less than the max value.


