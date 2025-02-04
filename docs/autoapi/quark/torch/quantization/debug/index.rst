:orphan:

:py:mod:`quark.torch.quantization.debug`
========================================

.. py:module:: quark.torch.quantization.debug


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.debug.weight_stats_hook
   quark.torch.quantization.debug.activation_stats_hook
   quark.torch.quantization.debug.distribution_plot
   quark.torch.quantization.debug.barplot
   quark.torch.quantization.debug.save_distribution_histogram
   quark.torch.quantization.debug.summarize_weight
   quark.torch.quantization.debug.summarize_activation
   quark.torch.quantization.debug.insert_stats_hooks
   quark.torch.quantization.debug.collect_quantization_statistics



.. py:function:: weight_stats_hook(module: quark.torch.quantization.tensor_quantize.FakeQuantizeBase, args: Tuple[Any, Ellipsis], output: torch.Tensor, module_name: str, log_dir: str, n_bins: int, stats: Dict[str, Any]) -> None

   Hook to collect statistics on the weight and bias quantization. This hook should only be attached to `FakeQuantizeBase` layers corresponding to weight and bias quantization.

   This hook should be attached with https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook.

   Args:
       module (FakeQuantizeBase): The torch.nn.Module this hook is being attached to.
       args (Tuple[Any, ...]): The module inputs, as specified in `torch.nn.Module.register_forward_hook` documentation.
       output (torch.Tensor): The module output, as specified in `torch.nn.Module.register_forward_hook` documentation.
       module_name (str): The module name, set with `functools.partial`. This is useful to access the module name from within the hook.
       log_dir (str): The directory the weight statistics will be saved to, set with `functools.partial`.
       n_bins (int): The number of bins inthe histograms of values that are saved for visualization.
       stats (Dict[str, Any]): The dictionary used to store statistics on the weight and bias quantization. It can be set using an empty handle using `functools.partial`. Passing a dictionary is useful to access outside of the hook its content that was modified from within the hook.


.. py:function:: activation_stats_hook(module: quark.torch.quantization.tensor_quantize.FakeQuantizeBase, args: Tuple[Any, Ellipsis], output: torch.Tensor, module_name: str, stats: Dict[str, Any]) -> None

   Hook to collect statistics on the activation quantization. This hook should only be attached to `FakeQuantizeBase` layers corresponding to input/output quantization.


.. py:function:: distribution_plot(histogram: Tuple[numpy.ndarray, numpy.ndarray], save_path: Union[str, pathlib.Path], title: str) -> None

   Plots and saves a bar plot using the bins and distribution from `histogram`. This is useful to save a given layer distribution, error, etc.


.. py:function:: barplot(labels: Collection[str], values: Iterable[float], name: str, log_dir: Union[str, pathlib.Path]) -> None

   Plots and saves a bar plot summary of values, each value having a label. This is useful to plot a summary of e.g. quantization error over many layers.


.. py:function:: save_distribution_histogram(module_name: str, tensor_stats: Dict[str, Any], log_dir: str) -> None

   Saves bar plots of activations. Utility function to be used by multiprocessing.


.. py:function:: summarize_weight(stats: Dict[str, Any], log_dir: pathlib.Path) -> None

   Saves a histogram of the distribution of the weight tensor for each weight tracked. Saves as well a summary plot of the L1 quantization error over all the different weight tensors.


.. py:function:: summarize_activation(stats: Dict[str, Any], log_dir: pathlib.Path) -> None

   Saves a summary over all activations of the error between the quantized / non-quantized model.


.. py:function:: insert_stats_hooks(model: torch.nn.Module, stats: Dict[str, Any], log_dir: pathlib.Path) -> Iterator[None]

   Inserts the hooks to track statistics about quantization error.


.. py:function:: collect_quantization_statistics(model: torch.nn.Module, dataloader: Optional[Union[torch.utils.data.DataLoader[torch.Tensor], torch.utils.data.DataLoader[List[Dict[str, torch.Tensor]]], torch.utils.data.DataLoader[Dict[str, torch.Tensor]], torch.utils.data.DataLoader[List[transformers.feature_extraction_utils.BatchFeature]]]], stats: Dict[str, Any], log_dir: pathlib.Path) -> None

   Collects (through the hooks attached to the model) statistics on the operators inputs/outputs to compute quantization error metrics, as well as on the weights.

   Moreover, this function writes to disk statistics, distribution and summary bar charts for the quantization of weights and activations.


