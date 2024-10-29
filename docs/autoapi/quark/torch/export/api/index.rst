:py:mod:`quark.torch.export.api`
================================

.. py:module:: quark.torch.export.api

.. autoapi-nested-parse::

   Quark Exporting API for PyTorch.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.api.ModelExporter



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.export.api.save_params
   quark.torch.export.api.import_model_info



.. py:class:: ModelExporter(config: quark.torch.export.config.config.ExporterConfig, export_dir: Union[pathlib.Path, str] = tempfile.gettempdir())


   Provides an API for exporting quantized Pytorch deep learning models.
   This class converts the quantized model to json-safetensors files or onnx graph, and saves to export_dir.

   Args:
       config (ExporterConfig): Configuration object containing settings for exporting.
       export_dir (Union[Path, str]): The target export directory. This could be a string or a pathlib.Path(string) object.

   .. py:method:: export_model_info(model: torch.nn.Module, model_type: str = '', model_dtype: torch.dtype = torch.float16, quant_config: Optional[quark.torch.quantization.config.config.Config] = None, export_type: Optional[str] = None, tokenizer: transformers.AutoTokenizer = None, custom_mode: str = 'quark') -> None

      This function aims to export json and safetensors files of the quantized Pytorch model.

      The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the safetensors file.

      Parameters:
          model (torch.nn.Module): The quantized model to be exported.
          model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.
          model_dtype (torch.dtype): The weight data type of the quantized model. Default is torch.float16.
          quant_config (Optional[Config]): Configuration object containing settings for quantization. Default is None.
          export_type (Optional[str]): The specific format in which the JSON and safetensors files are stored. Default is None. The file list of the default exporting format is the same as the original HuggingFace file list. On the basis of these files, add quantization information into them. If set to 'vllm-adopt', the exported files are customized for the VLLM compiler. This option is going to be deprecated soon.
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
              exporter.export_model_info(model, quant_config=quant_config)

          .. code-block:: python

              # vllm adopted exporting:
              export_path = "./output_dir"
              from quark.torch import ModelExporter
              from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
              NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                                         pack_method="reorder")
              export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
              exporter = ModelExporter(config=export_config, export_dir=export_path)
              exporter.export_model_info(model, model_type=model_type, model_dtype=model_dtype, export_type="vllm-adopt")

      Note:
          Currently, default exporting format supports large language models(LLM) in HuggingFace.
          If set to 'vllm-adopt', supported quantization types include fp8, int4_per_group, and w4a8_per_group, and supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.


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



.. py:function:: save_params(model: torch.nn.Module, model_type: str, args: Optional[Tuple[Any, Ellipsis]] = None, kwargs: Optional[Dict[str, Any]] = None, export_dir: Union[pathlib.Path, str] = tempfile.gettempdir(), quant_mode: quark.torch.quantization.config.type.QuantizationMode = QuantizationMode.eager_mode) -> None

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


.. py:function:: import_model_info(model: torch.nn.Module, model_info_dir: Union[pathlib.Path, str]) -> torch.nn.Module

   Instantiate a quantized large language model(LLM) from quark's json-safetensors exporting files.
   The json-safetensors files are exported using "export_model_info" API of ModelExporter class.

   Parameters:
       model (torch.nn.Module): The original HuggingFace large language model.
       model_info_dir (Union[Path, str]): The directory in which the quantized model files are stored.

   Returns:
       nn.Module: The reloaded quantized version of the input model. In this model, the weights of the quantized operators are stored in the real_quantized format.

   **Examples**:

       .. code-block:: python

           from quark.torch import import_model_info
           safetensors_model_dir = "./output_dir/json-safetensors"
           model = import_model_info(model, model_info_dir=safetensors_model_dir)

   Note:
       This function only supports large language models(LLM) of HuggingFace, and does not support dynamic quantized models for now.


