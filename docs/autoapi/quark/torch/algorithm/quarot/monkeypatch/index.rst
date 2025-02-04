:orphan:

:py:mod:`quark.torch.algorithm.quarot.monkeypatch`
==================================================

.. py:module:: quark.torch.algorithm.quarot.monkeypatch


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.quarot.monkeypatch.copy_func_with_new_globals
   quark.torch.algorithm.quarot.monkeypatch.add_wrapper_after_function_call_in_method



.. py:function:: copy_func_with_new_globals(f: Callable[Ellipsis, Any], globals: Optional[Dict[str, Any]] = None) -> Callable[Ellipsis, Any]

   Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)


.. py:function:: add_wrapper_after_function_call_in_method(module: torch.nn.Module, method_name: str, function_name: str, wrapper_fn: Callable[Ellipsis, Any]) -> Any

   This function adds a wrapper after the output of a function call in the method named `method_name`.
   Only calls directly in the method are affected. Calls by other functions called in the method are not affected.


