:orphan:

:py:mod:`quark.onnx.optimize`
=============================

.. py:module:: quark.onnx.optimize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.optimize.Optimize



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.optimize.optimize



.. py:class:: Optimize(model: onnx.ModelProto, op_types_to_quantize: List[str], nodes_to_quantize: Optional[List[str]], nodes_to_exclude: Optional[List[str]])




   A class for optimizations to be applied to onnx model before quantization.

   Args:
       model (onnx.ModelProto): The ONNX model to be optimized.
       op_types_to_quantize (list): A list of operation types to be quantized.
       nodes_to_quantize (list): A list of node names to be quantized.
       nodes_to_exclude (list): A list of node names to be excluded from quantization.


   .. py:method:: convert_bn_to_conv() -> None

      Convert BatchNormalization to Conv.
              


   .. py:method:: convert_reduce_mean_to_global_avg_pool() -> None

      Convert ReduceMean to GlobalAveragePool.
              


   .. py:method:: split_large_kernel_pool() -> None

      For pooling with an excessively large kernel size in the onnx model,
      split it into multiple smaller poolings.


   .. py:method:: convert_split_to_slice() -> None

      Convert Split to Slice.
              


   .. py:method:: fuse_instance_norm() -> None

      The split instance norm operation will be fused to InstanceNorm operation


   .. py:method:: fuse_l2_norm() -> None

      convert L2norm ops to LpNormalization


   .. py:method:: fold_batch_norm() -> None

      fold BatchNormalization to target operations


   .. py:method:: convert_clip_to_relu() -> None

      Convert Clip to Relu.


   .. py:method:: fold_batch_norm_after_concat() -> None

      fold BatchNormalization (after concat) to target operations



.. py:function:: optimize(model: onnx.ModelProto, op_types_to_quantize: List[str], nodes_to_quantize: Optional[List[str]], nodes_to_exclude: Optional[List[str]], convert_bn_to_conv: bool = True, convert_reduce_mean_to_global_avg_pool: bool = True, split_large_kernel_pool: bool = True, convert_split_to_slice: bool = True, fuse_instance_norm: bool = True, fuse_l2_norm: bool = True, fuse_gelu: bool = True, fuse_layer_norm: bool = True, fold_batch_norm: bool = True, convert_clip_to_relu: bool = True, fold_batch_norm_after_concat: bool = True, dedicate_dq_node: bool = False) -> onnx.ModelProto

   Optimize an ONNX model to meet specific constraints and requirements for deployment on an CPU/NPU.

   This function applies various optimization techniques to the provided ONNX model based on the specified parameters. The optimizations include fusing operations, converting specific layers, and folding batch normalization layers, among others.

   :param model: The ONNX model to be optimized.
   :type model: ModelProto
   :param op_types_to_quantize: List of operation types to be quantized.
   :type op_types_to_quantize: List[str]
   :param nodes_to_quantize: List of node names to explicitly quantize. If `None`, quantization is applied based on the operation types.
   :type nodes_to_quantize: Optional[List[str]]
   :param nodes_to_exclude: List of node names to exclude from quantization.
   :type nodes_to_exclude: Optional[List[str]]
   :param convert_bn_to_conv: Flag indicating whether to convert BatchNorm layers to Conv layers.
   :type convert_bn_to_conv: bool
   :param convert_reduce_mean_to_global_avg_pool: Flag indicating whether to convert ReduceMean layers to GlobalAveragePool layers.
   :type convert_reduce_mean_to_global_avg_pool: bool
   :param split_large_kernel_pool: Flag indicating whether to split large kernel pooling operations.
   :type split_large_kernel_pool: bool
   :param convert_split_to_slice: Flag indicating whether to convert Split layers to Slice layers.
   :type convert_split_to_slice: bool
   :param fuse_instance_norm: Flag indicating whether to fuse InstanceNorm layers.
   :type fuse_instance_norm: bool
   :param fuse_l2_norm: Flag indicating whether to fuse L2Norm layers.
   :type fuse_l2_norm: bool
   :param fuse_gelu: Flag indicating whether to fuse Gelu layers.
   :type fuse_gelu: bool
   :param fuse_layer_norm: Flag indicating whether to fuse LayerNorm layers.
   :type fuse_layer_norm: bool
   :param fold_batch_norm: Flag indicating whether to fold BatchNorm layers into preceding Conv layers.
   :type fold_batch_norm: bool
   :param convert_clip_to_relu: Flag indicating whether to convert Clip layers to ReLU layers.
   :type convert_clip_to_relu: bool
   :param fold_batch_norm_after_concat: Flag indicating whether to fold BatchNorm layers after concatenation operations.
   :type fold_batch_norm_after_concat: bool

   :return: The optimized ONNX model.
   :rtype: ModelProto

   Notes:
       - The `Optimize` class is used to apply the optimizations based on the provided flags.
       - The function returns the optimized model with the applied transformations.


