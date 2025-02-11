:py:mod:`quark.torch.export.api`
================================

.. py:module:: quark.torch.export.api

.. autoapi-nested-parse::

   Quark Exporting and Importing API for PyTorch.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.api.ModelExporter
   quark.torch.export.api.ModelImporter



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.export.api.save_params



.. py:class:: ModelExporter(config: quark.torch.export.config.config.ExporterConfig, export_dir: Union[pathlib.Path, str] = tempfile.gettempdir())


   Provides an API for exporting quantized Pytorch deep learning models.
   This class converts the quantized model to json-pth, json-safetensors files or onnx graph, and saves to export_dir.

   Args:
       config (ExporterConfig): Configuration object containing settings for exporting.
       export_dir (Union[Path, str]): The target export directory. This could be a string or a pathlib.Path(string) object.

   .. py:method:: export_quark_model(model: torch.nn.Module, quant_config: quark.torch.quantization.config.config.Config, custom_mode: str = 'quark') -> None

      This function aims to export json and pth files of the quantized Pytorch model by quark file format.
      The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the pth file.

      Parameters:
          model (transformers.PreTrainedModel): The quantized model to be exported.
          quant_config (Config): Configuration object containing settings for quantization. Default is None.
          custom_mode (str): Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:
              - `"quark"`: standard quark format. This is the default and recommended format that should be favored.
              - `"awq"`: targets AutoAWQ library.
              - `"fp8"`: targets vLLM-compatible fp8 models.

      Returns:
          None
      **Examples**:

          .. code-block:: python

              # default exporting:
              export_path = "./output_dir"
              from quark.torch import ModelExporter
              from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
              NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                         pack_method="reorder")
              export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
              exporter = ModelExporter(config=export_config, export_dir=export_path)
              quant_config = get_config(args.quant_scheme, args.group_size, args.model_dir, args.kv_cache_dtype, args.fp8_attention_quant, args.exclude_layers, args.pre_quantization_optimization, args.pre_optimization_config_file_path, args.quant_algo, args.quant_algo_config_file_path, model_type)
              exporter.export_quark_model(model, quant_config=quant_config, custom_mode=args.custom_mode)

      Note:
          Currently, default exporting quark format (json + pth).


   .. py:method:: get_export_model(model: torch.nn.Module, quant_config: quark.torch.quantization.config.config.Config, custom_mode: str = 'quark', add_export_info_for_hf: bool = True) -> torch.nn.Module

      Merges scales, replaces modules of the quantized model to prepare for export, and add export information in config.json.

      Scale merging selects the maximum scale value in specified `weight_group` as the scale for each module in the group.

      Build kv_scale selects the maximum kv_scale value in `kv_group` as the scale for the key projection output quantization and value projection output quantization.

      Module replacement converts the model's module (e.g. `QuantLinear`) according to the weight_format (to `QparamsLinear`).

      Parameters:
          model (transformers.PreTrainedModel): The quantized model to be exported.
          quant_config (Config): Configuration object containing settings for quantization.
          custom_mode (str): Whether to export the quantization config and model in a custom format expected by some downstream library. Possible options:
              - `"quark"`: standard quark format. This is the default and recommended format that should be favored.
              - `"awq"`: targets AutoAWQ library.
              - `"fp8"`: targets vLLM-compatible fp8 models.
      add_export_info_for_hf (bool): Whether to add export info of quark to config.json when using hf_format_export. When loading the model, we recover the kv_cache in autofp8 format through the weight file, but we need the name of kv_layer, it is very cumbersome to get it from quark's map, it is more reasonable to get it from config. If we find kv_scale in weight_flie and there is no special kv_layer_name, we will use k_proj,v_proj to recover kv_cache by default.


   .. py:method:: reset_model(model: torch.nn.Module) -> None

      Restore exported model to freezed Model for inferring, restore config content.


   .. py:method:: export_onnx_model(model: torch.nn.Module, input_args: Union[torch.Tensor, Tuple[float]], input_names: List[str] = [], output_names: List[str] = [], verbose: bool = False, opset_version: Optional[int] = None, do_constant_folding: bool = True, operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX, uint4_int4_flag: bool = False) -> None

      This function aims to export onnx graph of the quantized Pytorch model.

      Parameters:
          model (torch.nn.Module): The quantized model to be exported.
          input_args (Union[torch.Tensor, Tuple[float]]): Example inputs for this quantized model.
          input_names (List[str]): Names to assign to the input nodes of the onnx graph, in order. Default is empty list.
          output_names (List[str]): Names to assign to the output nodes of the onnx graph, in order. Default is empty list.
          verbose (bool): Flag to control showing verbose log or no. Default is False
          opset_version (Optional[int]): The version of the default (ai.onnx) opset to target. If not set, it will be valued the latest version that is stable for the current version of PyTorch.
          do_constant_folding (bool): Apply the constant-folding optimization. Default is False
          operator_export_type (torch.onnx.OperatorExportTypes): Export operator type in onnx graph. The choices include OperatorExportTypes.ONNX, OperatorExportTypes.ONNX_FALLTHROUGH, OperatorExportTypes.ONNX_ATEN and OperatorExportTypes.ONNX_ATEN_FALLBACK. Default is OperatorExportTypes.ONNX.
          uint4_int4_flag (bool): Flag to indicate uint4/int4 quantized model or not. Default is False.

      Returns:
          None

      **Examples**:

          .. code-block:: python

              from quark.torch import ModelExporter
              from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
              export_config = ExporterConfig(json_export_config=JsonExporterConfig())
              exporter = ModelExporter(config=export_config, export_dir=export_path)
              exporter.export_onnx_model(model, input_args)

      Note:
          Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
          In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.


   .. py:method:: export_gguf_model(model: torch.nn.Module, tokenizer_path: Union[str, pathlib.Path], model_type: str) -> None

      This function aims to export gguf file of the quantized Pytorch model.

      Parameters:
          model (torch.nn.Module): The quantized model to be exported.
          tokenizer_path (Union[str, Path]): Tokenizer needs to be encoded into gguf model. This argument specifies the directory path of tokenizer which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model
          model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.

      Returns:
          None

      **Examples**:

          .. code-block:: python

              from quark.torch import ModelExporter
              from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
              export_config = ExporterConfig(json_export_config=JsonExporterConfig())
              exporter = ModelExporter(config=export_config, export_dir=export_path)
              exporter.export_gguf_model(model, tokenizer_path, model_type)

      Note:
          Currently, only support asymetric int4 per_group weight-only quantization, and the group_size must be 32.
          Supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.



.. py:function:: save_params(model: torch.nn.Module, model_type: str, args: Optional[Tuple[Any, Ellipsis]] = None, kwargs: Optional[Dict[str, Any]] = None, export_dir: Union[pathlib.Path, str] = tempfile.gettempdir(), quant_mode: quark.torch.quantization.config.type.QuantizationMode = QuantizationMode.eager_mode, compressed: bool = False, reorder: bool = True) -> None

   Save the network architecture or configurations and parameters of the quantized model.
   For eager mode quantization, the model's configurations are stored in json file, and parameters including weight, bias, scale, and zero_point are stored in safetensors file.
   For fx_graph mode quantization, the model's network architecture and parameters are stored in pth file.

   Parameters:
       model (torch.nn.Module): The quantized model to be saved.
       model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.
       args (Optional[Tuple[Any, ...]]): Example tuple inputs for this quantized model. Only available for fx_graph mode quantization. Default is None.
       kwargs (Optional[Dict[str, Any]]): Example dict inputs for this quantized model. Only available for fx_graph mode quantization. Default is None.
       export_dir (Union[Path, str]): The target export directory. This could be a string or a pathlib.Path(string) object.
       quant_mode (QuantizationMode): The quantization mode. The choice includes "QuantizationMode.eager_mode" and "QuantizationMode.fx_graph_mode". Default is "QuantizationMode.eager_mode".
       compressed (bool): export the compressed (real quantized) model or QDQ model, Default is False and export the QDQ model
       reorder (bool): pack method, uses pack the weight(eg. packs four torch.int8 value into one torch.int32 value). Default is True

   Returns:
       None

   **Examples**:

       .. code-block:: python

           # eager mode:
           from quark.torch import save_params
           save_params(model, model_type=model_type, export_dir="./save_dir")

       .. code-block:: python

           # fx_graph mode:
           from quark.torch.export.api import save_params
           save_params(model,
                       model_type=model_type,
                       args=example_inputs,
                       export_dir="./save_dir",
                       quant_mode=QuantizationMode.fx_graph_mode)


.. py:class:: ModelImporter(model_info_dir: str)


   Provides an API for importing quantized Pytorch deep learning models.
   This class load json-pth or json-safetensors files to model.

   Args:
       model_info_dir (str): The target import directory.

   .. py:method:: import_model_info(model: torch.nn.Module) -> torch.nn.Module

      This function aims to import quark(json-pth) files of the HuggingFace large language model.

      It could recover the weight, bias, scale, and zeropoint information of the model and execute the inference

      Parameters:
          model (transformers.PreTrainedModel): The original HuggingFace large language model.

      Returns:
          model: Models that have completed weight import
      **Examples**:

          .. code-block:: python

              # default exporting:
              import_model_dir = "./import_model_dir"
              from quark.torch import ModelImporter
              importer = ModelImporter(model_info_dir=args.import_model_dir)
              model = importer.import_model_info(model)



   .. py:method:: import_model(model: torch.nn.Module, model_config: quark.torch.export.main_import.pretrained_config.PretrainedConfig, model_state_dict: Dict[str, Any]) -> torch.nn.Module

      This function uses the loaded state_dict and config to build the model



