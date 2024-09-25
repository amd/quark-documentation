:orphan:

:py:mod:`quark.torch.quantization.graph.torch_utils`
====================================================

.. py:module:: quark.torch.quantization.graph.torch_utils


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.torch_utils.is_conv1d_node
   quark.torch.quantization.graph.torch_utils.is_conv2d_node
   quark.torch.quantization.graph.torch_utils.is_conv3d_node
   quark.torch.quantization.graph.torch_utils.is_batchnorm2d_node
   quark.torch.quantization.graph.torch_utils.is_dropout_node
   quark.torch.quantization.graph.torch_utils.allow_exported_model_train_eval



.. py:function:: is_conv1d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv1d op.


.. py:function:: is_conv2d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv2d op.


.. py:function:: is_conv3d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv3d op.


.. py:function:: is_batchnorm2d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten batch_norm op.


.. py:function:: is_dropout_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten dropout op.


.. py:function:: allow_exported_model_train_eval(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   Allow users to call `model.train()` and `model.eval()` on GraphModule,
   the effect of changing behavior between the two modes limited to special ops only,
     which are currently dropout and batchnorm.

   Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
   does in eager models, but only provides an approximation.



