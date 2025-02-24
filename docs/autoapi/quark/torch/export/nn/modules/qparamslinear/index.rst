:orphan:

:py:mod:`quark.torch.export.nn.modules.qparamslinear`
=====================================================

.. py:module:: quark.torch.export.nn.modules.qparamslinear


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.nn.modules.qparamslinear.QparamsOperator
   quark.torch.export.nn.modules.qparamslinear.QParamsLinear




.. py:class:: QparamsOperator(*args: Any, **kwargs: Any)




   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
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


.. py:class:: QParamsLinear(linear: torch.nn.Linear, custom_mode: str, pack_method: Optional[str] = 'reorder', quant_config: Optional[quark.torch.quantization.config.config.QuantizationConfig] = None)




   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
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

   .. py:method:: from_module(linear: torch.nn.Linear, custom_mode: str, pack_method: Optional[str] = 'reorder', quant_config: Optional[quark.torch.quantization.config.config.QuantizationConfig] = None) -> QParamsLinear
      :classmethod:

      Build a QParamsLinear from a QuantLinear or nn.Linear.
      Initialize the shape and data type of weight and bias in importing.
      Initialize weight and bias in exporting.


   .. py:method:: forward(*args: Any, **kwargs: Any) -> torch.Tensor

      Dequantizes quantized weight/bias, runs a linear in high precision and apply QDQ on the (input)activation/output if required.


   .. py:method:: pack_qinfo() -> None

      Calls `RealQuantizer.pack_zero_point`` and `RealQuantizer.maybe_transpose_scale` to do scale, zero_point packing if required.


   .. py:method:: state_dict(*args: Any, destination: Any = None, prefix: str = '', keep_vars: bool = False) -> Any

      Return a dictionary containing references to the whole state of the module.

      Both parameters and persistent buffers (e.g. running averages) are
      included. Keys are corresponding parameter and buffer names.
      Parameters and buffers set to ``None`` are not included.

      .. note::
          The returned object is a shallow copy. It contains references
          to the module's parameters and buffers.

      .. warning::
          Currently ``state_dict()`` also accepts positional arguments for
          ``destination``, ``prefix`` and ``keep_vars`` in order. However,
          this is being deprecated and keyword arguments will be enforced in
          future releases.

      .. warning::
          Please avoid the use of argument ``destination`` as it is not
          designed for end-users.

      Args:
          destination (dict, optional): If provided, the state of module will
              be updated into the dict and the same object is returned.
              Otherwise, an ``OrderedDict`` will be created and returned.
              Default: ``None``.
          prefix (str, optional): a prefix added to parameter and buffer
              names to compose the keys in state_dict. Default: ``''``.
          keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
              returned in the state dict are detached from autograd. If it's
              set to ``True``, detaching will not be performed.
              Default: ``False``.

      Returns:
          dict:
              a dictionary containing a whole state of the module

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> module.state_dict().keys()
          ['bias', 'weight']




