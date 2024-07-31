:orphan:

:py:mod:`quark.onnx.simulate_dpu`
=================================

.. py:module:: quark.onnx.simulate_dpu


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.simulate_dpu.simulate_transforms



.. py:function:: simulate_transforms(model: onnx.ModelProto, should_quantize_node: Callable[[Any], bool], nodes_to_quantize: List[str], nodes_to_exclude: List[str], convert_leaky_relu_to_dpu_version: bool = True, convert_sigmoid_to_hard_sigmoid: bool = True, convert_hard_sigmoid_to_dpu_version: bool = True, convert_avg_pool_to_dpu_version: bool = True, convert_reduce_mean_to_dpu_version: bool = True, convert_softmax_to_dpu_version: bool = True, convert_instance_norm_to_dpu_version: bool = True) -> tuple[onnx.ModelProto, List[str]]

   Transforming models to meet the DPU constraints.


