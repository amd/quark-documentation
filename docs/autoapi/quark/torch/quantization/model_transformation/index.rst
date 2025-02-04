:orphan:

:py:mod:`quark.torch.quantization.model_transformation`
=======================================================

.. py:module:: quark.torch.quantization.model_transformation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.model_transformation.process_model_transformation
   quark.torch.quantization.model_transformation.setup_config_per_layer
   quark.torch.quantization.model_transformation.in_place_replace_layer



.. py:function:: process_model_transformation(model: torch.nn.Module, config: quark.torch.quantization.config.config.Config) -> torch.nn.Module

   Replaces modules to be quantized by their quantized equivalent (e.g. nn.Linear by QuantLinear), based on the provided global `config`.


.. py:function:: setup_config_per_layer(config: quark.torch.quantization.config.config.Config, named_modules: Dict[str, torch.nn.Module], module_configs: Dict[str, quark.torch.quantization.config.config.QuantizationConfig]) -> None

   Retrieves the `QuantizationConfig` used for each layer, based on the
   `config`'s `global_quant_config`, `layer_quant_config` and `layer_type_quant_config`.


.. py:function:: in_place_replace_layer(model: torch.nn.Module, config: quark.torch.quantization.config.config.Config, named_modules: Dict[str, torch.nn.Module], module_configs: Dict[str, quark.torch.quantization.config.config.QuantizationConfig]) -> None

   Replaces `nn.Linear`, `nn.Conv2d`, etc. marked for quantization in `module_configs` by their quantized module equivalent.


