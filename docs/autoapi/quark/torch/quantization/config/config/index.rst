:py:mod:`quark.torch.quantization.config.config`
================================================

.. py:module:: quark.torch.quantization.config.config

.. autoapi-nested-parse::

   Quark Quantization Config API for PyTorch



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.config.config.Config
   quark.torch.quantization.config.config.QuantizationConfig
   quark.torch.quantization.config.config.QuantizationSpec
   quark.torch.quantization.config.config.SmoothQuantConfig
   quark.torch.quantization.config.config.RotationConfig
   quark.torch.quantization.config.config.AWQConfig
   quark.torch.quantization.config.config.GPTQConfig




.. py:class:: Config


   A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

   :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
   :param Dict[str, QuantizationConfig] layer_type_quant_config: A dictionary mapping from layer types (e.g., nn.Conv2d, nn.Linear) to their quantization configurations.
   :param Dict[str, QuantizationConfig] layer_quant_config: A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization. Default is an empty dictionary.
   :param List[str] exclude: A list of layer names to be excluded from quantization, enabling selective quantization of the model. Default is an empty list.
   :param Optional[AlgoConfig] algo_config: Optional configuration for the quantization algorithm, such as GPTQ and AWQ. After this process, the datatype/fake_datatype of weights will be changed with quantization scales. Default is None.
   :param QuantizationMode quant_mode: The quantization mode to be used (eager_mode or fx_graph_mode). Default is eager_mode.
   :param List[PreQuantOptConfig] pre_quant_opt_config: Optional pre-processing optimization, such as Equalization and SmoothQuant. After this process, the value of weights will be changed, but the dtype/fake_dtype will be the same. Default is an empty list.
   :param Optional[int] log_severity_level: 0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is 1.


.. py:class:: QuantizationConfig


   A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

   :param Optional[QuantizationSpec] input_tensors: Input tensors quantization specification. If None, following the hierarchical quantization setup. e.g. If the input_tensors in layer_type_quant_config is None, the configuration from global_quant_config will be used instead. Defaults to None. If None in global_quant_config, input_tensors are not quantized.
   :param Optional[QuantizationSpec] output_tensors: Output tensors quantization specification. Defaults to None. If None, the same as above.
   :param Optional[QuantizationSpec] weight: The weights tensors quantization specification. Defaults to None. If None, the same as above.
   :param Optional[QuantizationSpec] bias: The bias tensors quantization specification. Defaults to None. If None, the same as above.
   :param Optional[DeviceType] target_device: Configuration specifying the target device (e.g., CPU, GPU, IPU) for the quantized model.



.. py:class:: QuantizationSpec


   A data class that defines the specifications for quantizing tensors within a model.

   :param Dtype dtype: The data type for quantization (e.g., int8, int4).
   :param Optional[bool] is_dynamic: Specifies whether dynamic or static quantization should be used. Default is None, which indicates no specification.
   :param Optional[Type[ObserverBase]] observer_cls: The class of observer to be used for determining quantization parameters like min/max values. Default is None.
   :param Optional[QSchemeType] qscheme: The quantization scheme to use, such as per_tensor, per_channel or per_group. Default is None.
   :param Optional[int] ch_axis: The channel axis for per-channel quantization. Default is None.
   :param Optional[int] group_size: The size of the group for per-group quantization, also the block size for MX datatypes. Default is None.
   :param Optional[bool] symmetric: Indicates if the quantization should be symmetric around zero. If True, quantization is symmetric. If None, it defers to a higher-level or global setting. Default is None.
   :param Optional[RoundType] round_method: The rounding method during quantization, such as half_even. If None, it defers to a higher-level or default method. Default is None.
   :param Optional[ScaleType] scale_type: Defines the scale type to be used for quantization, like power of two or float. If None, it defers to a higher-level setting or uses a default method. Default is None.
   :param Optional[Dtype] mx_element_dtype: Defines the data type to be used for the element type when using mx datatypes, the shared scale effectively uses FP8 E8M0.


.. py:class:: SmoothQuantConfig




   A data class that defines the specifications for Smooth Quantization.

   :param str name: The name of the configuration, typically used to identify different quantization settings. Default is "smoothquant".
   :param int alpha: The factor of adjustment in the quantization formula, influencing how aggressively weights are quantized. Default is 1.
   :param float scale_clamp_min: The minimum scaling factor to be used during quantization, preventing the scale from becoming too small. Default is 1e-3.
   :param Optional[List[Dict[str, str]]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is None.
   :param Optional[str] model_decoder_layers: Specifies any particular decoder layers in the model that might have unique quantization requirements. Default is None.
   :param type[SmoothQuantProcessor] processor: The processor object that applies the algorithm.


.. py:class:: RotationConfig




   A data class that defines the specifications for rotation settings in processing algorithms.

   :param str name: The name of the configuration, typically used to identify different rotation settings. Default is "rotation".
   :param bool random: A boolean flag indicating whether the rotation should be applied randomly. This can be useful for data augmentation purposes where random rotations may be required. Default is False.
   :param type[RotationProcessor] processor: The processor object that applies the rotation algorithm, defaulting to RotationProcessor which handles the actual rotation logic.


.. py:class:: AWQConfig




   Configuration for Activation-aware Weight Quantization (AWQ).

   :param str name: The name of the quantization configuration. Default is "awq".
   :param type[AwqProcessor] processor: The processor type that handles the AWQ algorithm logic.
   :param Optional[List[Dict[str, str]]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is None.
   :param Optional[str] model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is None.


.. py:class:: GPTQConfig




   A data class that defines the specifications for Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ).

   :param str name: The configuration name. Default is "gptq".
   :param float damp_percent: The percentage used to dampen the quantization effect, aiding in the maintenance of accuracy post-quantization. Default is 0.01.
   :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is True.
   :param bool static_groups: Specifies whether the order of groups for quantization are static or can be dynamically adjusted. Default is True. Quark export only support static_groups as True.
   :param bool true_sequential: Indicates whether the quantization should be applied in a truly sequential manner across the layers. Default is True.
   :param Optional[List[str]] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling. Default is None.
   :param Optional[str] model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model. Default is None.
   :param type[GptqProcessor] processor: Processor of the algorithm.


