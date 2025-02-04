:py:mod:`quark.torch.export.config.config`
==========================================

.. py:module:: quark.torch.export.config.config

.. autoapi-nested-parse::

   Quark Exporting Config API for PyTorch



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.config.config.ExporterConfig
   quark.torch.export.config.config.JsonExporterConfig
   quark.torch.export.config.config.OnnxExporterConfig




.. py:class:: ExporterConfig


   A class that encapsulates comprehensive exporting configurations for a machine learning model, allowing for detailed control over exporting parameters across different exporting formats.

   :param Optional[JsonExporterConfig] json_export_config: Global configuration for json-safetensors exporting.
   :param Optional[OnnxExporterConfig] onnx_export_config: Global configuration onnx exporting. Default is None.


.. py:class:: JsonExporterConfig


   A data class that specifies configurations for json-safetensors exporting.

   :param Optional[List[List[str]]] weight_merge_groups: A list of operators group that share the same weight scaling factor. These operators' names should correspond to the original module names from the model. Additionally, wildcards can be used to denote a range of operators. Default is None.
   :param List[str] kv_cache_group: A list of operators group that should be merged to kv_cache. These operators' names should correspond to the original module names from the model. Additionally, wildcards can be used to denote a range of operators.
   :param str weight_format: The flag indicating whether to export the real quantized weights.
   :param str pack_method: The flag indicating whether to reorder the quantized tensors.



.. py:class:: OnnxExporterConfig


   A data class that specifies configurations for onnx exporting.


