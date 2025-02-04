:orphan:

:py:mod:`quark.torch.export.utils`
==================================

.. py:module:: quark.torch.export.utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.export.utils.preprocess_import_info
   quark.torch.export.utils.split_params_for_DbrxExperts



.. py:function:: preprocess_import_info(model_state_dict: Dict[str, torch.Tensor], is_kv_cache: bool, kv_layers_name: Optional[List[str]], custom_mode: str) -> tuple[Dict[str, Any], bool, Optional[List[str]]]

   Load model weights, preprocess state_dict for some cases such as dbrx split, fp8 kv_cache, tied_parameter, etc.


.. py:function:: split_params_for_DbrxExperts(model_state_dict: Dict[str, Any], dbrx_experts_groups: List[List[str]]) -> None

   The moe part of dbrx needs special treatment, when loading a model, we do some splitting of that model, so the tensor that is loaded in here, needs to be split as well


