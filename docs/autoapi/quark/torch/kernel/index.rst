:orphan:

:py:mod:`quark.torch.kernel`
============================

.. py:module:: quark.torch.kernel


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.kernel.QuantE4M3Function
   quark.torch.kernel.DequantE4M3Function
   quark.torch.kernel.QuantE5M2Function
   quark.torch.kernel.DequantE5M2Function
   quark.torch.kernel.ScaledFakeQuantizeFunction
   quark.torch.kernel.NonScaledFakeQuantizeFunction
   quark.torch.kernel.ScaledRealQuantizeFunction
   quark.torch.kernel.NonScaledRealQuantizeFunction
   quark.torch.kernel.DeQuantizeFunction
   quark.torch.kernel.TQTQuantize
   quark.torch.kernel.LSQQuantize




.. py:class:: QuantE4M3Function(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: DequantE4M3Function(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: QuantE5M2Function(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: DequantE5M2Function(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: ScaledFakeQuantizeFunction(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Optional[torch.Tensor], axis: Optional[int], group_size: Optional[int], quant_min: Union[int, float], quant_max: Union[int, float], round_mode: Optional[int], qscheme: Optional[str], mx_element_dtype: Optional[str]) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: NonScaledFakeQuantizeFunction(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int, block_size: int) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: ScaledRealQuantizeFunction(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor, None], axis: Union[int, None], group_size: Union[int, None], quant_min: Union[int, float], quant_max: Union[int, float], round_mode: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: NonScaledRealQuantizeFunction(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int, block_size: int) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: DeQuantizeFunction(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor, None], axis: Union[int, None], group_size: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: TQTQuantize(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, inputs: torch.Tensor, log_threshold: torch.Tensor, zero_point: torch.Tensor, domain: torch.Tensor, round_mode: int) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_outputs: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



.. py:class:: LSQQuantize(*args, **kwargs)




   Base class to create custom `autograd.Function`.

   To create a custom `autograd.Function`, subclass this class and implement
   the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
   op in the forward pass, call the class method ``apply``. Do not call
   :meth:`forward` directly.

   To ensure correctness and best performance, make sure you are calling the
   correct methods on ``ctx`` and validating your backward function using
   :func:`torch.autograd.gradcheck`.

   See :ref:`extending-autograd` for more details on how to use this class.

   Examples::

       >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
       >>> class Exp(Function):
       >>>     @staticmethod
       >>>     def forward(ctx, i):
       >>>         result = i.exp()
       >>>         ctx.save_for_backward(result)
       >>>         return result
       >>>
       >>>     @staticmethod
       >>>     def backward(ctx, grad_output):
       >>>         result, = ctx.saved_tensors
       >>>         return grad_output * result
       >>>
       >>> # Use it by calling the apply method:
       >>> # xdoctest: +SKIP
       >>> output = Exp.apply(input)

   .. py:method:: forward(ctx: Any, x: torch.tensor, scale: torch.tensor, zero_point: torch.tensor, grad_factor: float, quant_min: int, quant_max: int, ch_axis: Optional[int] = None, rounding_key: int = 8) -> Any
      :staticmethod:

      Define the forward of the custom autograd Function.

      This function is to be overridden by all subclasses.
      There are two ways to define forward:

      Usage 1 (Combined forward and ctx)::

          @staticmethod
          def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
              pass

      - It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).
      - See :ref:`combining-forward-context` for more details

      Usage 2 (Separate forward and ctx)::

          @staticmethod
          def forward(*args: Any, **kwargs: Any) -> Any:
              pass

          @staticmethod
          def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
              pass

      - The forward no longer accepts a ctx argument.
      - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
        staticmethod to handle setting up the ``ctx`` object.
        ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
        to the forward.
      - See :ref:`extending-autograd` for more details

      The context can be used to store arbitrary data that can be then
      retrieved during the backward pass. Tensors should not be stored
      directly on `ctx` (though this is not currently enforced for
      backward compatibility). Instead, tensors should be saved either with
      :func:`ctx.save_for_backward` if they are intended to be used in
      ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
      if they are intended to be used for in ``jvp``.


   .. py:method:: backward(ctx: Any, grad_output: torch.Tensor) -> Any
      :staticmethod:

      Define a formula for differentiating the operation with backward mode automatic differentiation.

      This function is to be overridden by all subclasses.
      (Defining this function is equivalent to defining the ``vjp`` function.)

      It must accept a context :attr:`ctx` as the first argument, followed by
      as many outputs as the :func:`forward` returned (None will be passed in
      for non tensor outputs of the forward function),
      and it should return as many tensors, as there were inputs to
      :func:`forward`. Each argument is the gradient w.r.t the given output,
      and each returned value should be the gradient w.r.t. the
      corresponding input. If an input is not a Tensor or is a Tensor not
      requiring grads, you can just pass None as a gradient for that input.

      The context can be used to retrieve tensors saved during the forward
      pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
      of booleans representing whether each input needs gradient. E.g.,
      :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
      first input to :func:`forward` needs gradient computed w.r.t. the
      output.



