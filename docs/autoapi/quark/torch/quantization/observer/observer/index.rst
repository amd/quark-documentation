:orphan:

:py:mod:`quark.torch.quantization.observer.observer`
====================================================

.. py:module:: quark.torch.quantization.observer.observer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.observer.observer.ObserverBase
   quark.torch.quantization.observer.observer.PlaceholderObserver
   quark.torch.quantization.observer.observer.UniformScalingObserver
   quark.torch.quantization.observer.observer.PerTensorMinMaxObserver
   quark.torch.quantization.observer.observer.PerChannelMinMaxObserver
   quark.torch.quantization.observer.observer.PerBlockMXObserver
   quark.torch.quantization.observer.observer.PerBlockBFPObserver
   quark.torch.quantization.observer.observer.PerGroupMinMaxObserver
   quark.torch.quantization.observer.observer.PerTensorHistogramObserver
   quark.torch.quantization.observer.observer.PerTensorHistogramObserverPro
   quark.torch.quantization.observer.observer.PerTensorPercentileObserver
   quark.torch.quantization.observer.observer.PerTensorMSEObserver




.. py:class:: ObserverBase(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: PlaceholderObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer only passes its configuration to the quantized module's ``.from_float()``.

   Does not have any calculation.

   Only can be used for quantization to float16 and bfloat16 which doesn't require determining
   ranges.

   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



.. py:class:: UniformScalingObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, eps: float = torch.finfo(torch.float32).eps)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: calculate_qparams(min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]

      Calculates the quantization parameters.


   .. py:method:: extra_repr() -> str

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.


   .. py:method:: reset_min_max_vals() -> None

      Resets the min/max values.



.. py:class:: PerTensorMinMaxObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: forward(x_orig: torch.Tensor) -> torch.Tensor

      Records the running minimum and maximum of ``x``.



.. py:class:: PerChannelMinMaxObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, eps: float = torch.finfo(torch.float32).eps)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.



.. py:class:: PerBlockMXObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, eps: float = torch.finfo(torch.float32).eps)




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: PerBlockBFPObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, eps: float = torch.finfo(torch.float32).eps)




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: PerGroupMinMaxObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, eps: float = torch.finfo(torch.float32).eps)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: calculate_qparams(min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]

      Calculates the quantization parameters.



.. py:class:: PerTensorHistogramObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: forward(x_orig: torch.Tensor) -> torch.Tensor

      Records the running histogram of ``x_orig``.

      Raises:
      - ValueError: If the `self.symmetric` argument is False.




.. py:class:: PerTensorHistogramObserverPro(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None, bins: int = 256, reduce_range: bool = False, upsample_rate: int = 384)




   A wrap of pytorch version observer: HistogramObserver


.. py:class:: PerTensorPercentileObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: get_min_max_by_percentile(histogram: torch.Tensor, bin_edges: torch.Tensor, percentile: float) -> Tuple[torch.Tensor, torch.Tensor]

      Calculate the minimum and maximum values of a histogram at a specified percentile.

      Parameters:
      - histogram (torch.Tensor): A tensor representing the histogram of the data. Each element
      in the histogram represents the frequency of data in the corresponding bin.
      - bin_edges (torch.Tensor): A tensor containing the edge values that correspond to the
      bins represented in the histogram. There should be one more element in `bin_edges` than
      in `histogram`.
      - percentile (int): The percentile at which to determine the minimum and maximum values.
      The value should be an integer between 0 and 100.

      Returns:
      - Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors. The first tensor
      is the value at the specified percentile, and the second tensor is the value at the
      complementary percentile (i.e., 100-percentile).

      Raises:
      - ValueError: If the `percentile` argument is not within the range 0 to 100.



.. py:class:: PerTensorMSEObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: get_min_max_by_mse(calib_hist: torch.Tensor, calib_bin_edges: torch.Tensor, stride: int = 1, start_bin: int = 2045) -> Tuple[torch.Tensor, torch.Tensor]

      Returns amax that minimizes MSE of the collected histogram.



