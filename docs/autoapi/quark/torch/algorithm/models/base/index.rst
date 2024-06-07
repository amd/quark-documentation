:orphan:

:py:mod:`quark.torch.algorithm.models.base`
===========================================

.. py:module:: quark.torch.algorithm.models.base


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.models.base.BaseQuantModelForCausalLM




.. py:class:: BaseQuantModelForCausalLM(model: transformers.PreTrainedModel, is_quantized: bool, config: transformers.PretrainedConfig, algo_config: quark.torch.quantization.config.config.AlgoConfig)




   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool


