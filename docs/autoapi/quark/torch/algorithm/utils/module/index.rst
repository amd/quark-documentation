:orphan:

:py:mod:`quark.torch.algorithm.utils.module`
============================================

.. py:module:: quark.torch.algorithm.utils.module


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.utils.module.get_nested_attr_from_module



.. py:function:: get_nested_attr_from_module(obj: torch.nn.Module, attr_path: str) -> Any

   Retrieves the value of a nested attribute based on a given attribute path string.

   Parameters:
   - obj: The starting object.
   - attr_path: The string representing the attribute path, such as "model.decoder.layers".

   Returns:
   - The value of the nested attribute.


