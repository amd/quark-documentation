:orphan:

:py:mod:`quark.torch.quantization.utils`
========================================

.. py:module:: quark.torch.quantization.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.utils.set_op_by_name
   quark.torch.quantization.utils.t_exponent



.. py:function:: set_op_by_name(layer: Union[torch.nn.Module, torch.nn.ModuleList], name: str, new_module: torch.nn.Module) -> None

   Replaces a submodule in a given neural network layer with a new module(e.g. quantized module). The submodule to be
   replaced is identified by the 'name' parameter, which specifies the name of the submodule
   using dot notation. If the name includes dots, it navigates through nested submodules
   to find the specific layer to replace. Otherwise, it directly replaces the submodule in the
   provided layer.

   Parameters:
   - layer: The top-level module containing the submodule.
   - name: name of the submodule, split by dots.
   - new_module: The new module to replace the existing one, for example the quantized module.


.. py:function:: t_exponent(t: torch.Tensor) -> torch.Tensor

   Get element exponents

   Args:
       t (torch.Tensor): Input tensor

   Returns:
       torch.Tensor: Exponents for each elements. NaN and Inf are treated as zeros.



