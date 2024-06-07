:orphan:

:py:mod:`quark.torch.quantization.tensor_quantize`
==================================================

.. py:module:: quark.torch.quantization.tensor_quantize


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.tensor_quantize.FakeQuantizeBase
   quark.torch.quantization.tensor_quantize.FakeQuantize
   quark.torch.quantization.tensor_quantize.FreezedFakeQuantize
   quark.torch.quantization.tensor_quantize.SequentialFakeQuantize




.. py:class:: FakeQuantizeBase




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.



.. py:class:: FakeQuantize(quant_spec: quark.torch.quantization.config.config.QuantizationSpec, **kwargs: Any)




   Base fake quantize module.

   Base fake quantize module
   Any fake quantize implementation should derive from this class.

   Concrete fake quantize module should follow the same API. In forward, they will update
   the statistics of the observed Tensor and fake quantize the input. They should also provide a
   `calculate_qparams` function that computes the quantization parameters given
   the collected statistics.


   .. py:property:: block_sizes
      :type: int

      Return block_sizes for quantization.

   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.


   .. py:method:: export_amax() -> Optional[torch.Tensor]

      Adapter for GPU export



.. py:class:: FreezedFakeQuantize




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


.. py:class:: SequentialFakeQuantize(*args: torch.nn.modules.module.Module)
              SequentialFakeQuantize(arg: OrderedDict[str, Module])




   A sequential container.

   Modules will be added to it in the order they are passed in the
   constructor. Alternatively, an ``OrderedDict`` of modules can be
   passed in. The ``forward()`` method of ``Sequential`` accepts any
   input and forwards it to the first module it contains. It then
   "chains" outputs to inputs sequentially for each subsequent module,
   finally returning the output of the last module.

   The value a ``Sequential`` provides over manually calling a sequence
   of modules is that it allows treating the whole container as a
   single module, such that performing a transformation on the
   ``Sequential`` applies to each of the modules it stores (which are
   each a registered submodule of the ``Sequential``).

   What's the difference between a ``Sequential`` and a
   :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
   sounds like--a list for storing ``Module`` s! On the other hand,
   the layers in a ``Sequential`` are connected in a cascading way.

   Example::

       # Using Sequential to create a small model. When `model` is run,
       # input will first be passed to `Conv2d(1,20,5)`. The output of
       # `Conv2d(1,20,5)` will be used as the input to the first
       # `ReLU`; the output of the first `ReLU` will become the input
       # for `Conv2d(20,64,5)`. Finally, the output of
       # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
       model = nn.Sequential(
                 nn.Conv2d(1,20,5),
                 nn.ReLU(),
                 nn.Conv2d(20,64,5),
                 nn.ReLU()
               )

       # Using Sequential with OrderedDict. This is functionally the
       # same as the above code
       model = nn.Sequential(OrderedDict([
                 ('conv1', nn.Conv2d(1,20,5)),
                 ('relu1', nn.ReLU()),
                 ('conv2', nn.Conv2d(20,64,5)),
                 ('relu2', nn.ReLU())
               ]))


