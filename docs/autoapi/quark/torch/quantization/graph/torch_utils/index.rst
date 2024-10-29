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
   quark.torch.quantization.graph.torch_utils.is_convtranspose2d_node
   quark.torch.quantization.graph.torch_utils.is_batchnorm2d_node
   quark.torch.quantization.graph.torch_utils.is_dropout_node
   quark.torch.quantization.graph.torch_utils.is_cat_node
   quark.torch.quantization.graph.torch_utils.allow_exported_model_train_eval



Attributes
~~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.graph.torch_utils.CAT_OPS
   quark.torch.quantization.graph.torch_utils.BATCHNORM_OPS


.. py:data:: CAT_OPS

   # the possible batchnorm ops that parse from nn.BatchNorm2d()
   # NOTE: from PyTorch official doc, the bn operation will be unified in the future and will not have so many version
   # /pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml


.. py:data:: BATCHNORM_OPS

   batch_norm:
       (input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled) -> Tensor
   cudnn_batch_norm:
       (input, weight, bias, running_mean, running_var, training, momentum, epsilon) -> (Tensor, Tensor, Tensor, Tensor)
   native_batch_norm:
       (input, weight, bias, running_mean, running_var, training, momentum, eps) -> (Tensor, Tensor, Tensor)
   _native_batch_norm_legit:
       (input, weight, bias, running_mean, running_var, training, momentum, eps) -> (Tensor, Tensor, Tensor)
   miopen_batch_norm
       (input, weight, bias, running_mean, running_var, training, momentum, epsilon) -> (Tensor, Tensor, Tensor)
   _native_batch_norm_legit_no_training
       (input, weight, bias, running_mean, running_var, momentum, eps) -> (Tensor, Tensor, Tensor)


.. py:function:: is_conv1d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv1d op.


.. py:function:: is_conv2d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv2d op.


.. py:function:: is_conv3d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv3d op.


.. py:function:: is_convtranspose2d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten conv_transpose2d op.


.. py:function:: is_batchnorm2d_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten batch_norm op.


.. py:function:: is_dropout_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten dropout op.


.. py:function:: is_cat_node(n: torch.fx.Node) -> bool

   Return whether the node refers to an aten cat op.


.. py:function:: allow_exported_model_train_eval(model: torch.fx.GraphModule) -> torch.fx.GraphModule

   Allow users to call `model.train()` and `model.eval()` on GraphModule,
   the effect of changing behavior between the two modes limited to special ops only,
     which are currently dropout and batchnorm.

   Note: This does not achieve the same effect as what `model.train()` and `model.eval()`
   does in eager models, but only provides an approximation.



