:orphan:

:py:mod:`quark.onnx.equalization`
=================================

.. py:module:: quark.onnx.equalization


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.equalization.CLE_PAIR_TYPE
   quark.onnx.equalization.Equalization



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.equalization.cle_transforms



.. py:class:: CLE_PAIR_TYPE




   Generic enumeration.

   Derive from this class to define new enumerations.


.. py:class:: Equalization




   A class for layers equalization
   Args:
       model (onnx.ModelProto): The ONNX model to be optimized.
       op_types_to_quantize (list): A list of operation types to be quantized.
       nodes_to_quantize (list): A list of node names to be quantized.
       nodes_to_exclude (list): A list of node names to be excluded from quantization.



.. py:function:: cle_transforms(model: onnx.ModelProto, op_types_to_quantize: List[str], nodes_to_quantize: List[str], nodes_to_exclude: List[str], cle_steps: int = -1, cle_balance_method: str = 'max', cle_weight_threshold: float = 0.5, cle_scale_append_bias: bool = True, cle_scale_use_threshold: bool = True, cle_total_layer_diff_threshold: float = 1.9e-07) -> Any

   Equanlization transform models.


