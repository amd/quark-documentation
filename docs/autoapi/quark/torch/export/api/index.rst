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




.. py:class:: ModelExporter(config: quark.torch.export.config.config.ExporterConfig, export_dir: Union[pathlib.Path, str] = tempfile.gettempdir())


   Provides an API for exporting quantized Pytorch deep learning models.
   This class converts the quantized model to json-safetensors files or onnx graph, and saves to export_dir.

   :param config: Configuration object containing settings for exporting.
   :type config: ExporterConfig
   :param export_dir: The target export directory. This could be a string or a pathlib.Path(string) object.
   :type export_dir: Union[Path, str]

   .. py:method:: export_model_info(model: torch.nn.Module, model_type: str = '', model_dtype: torch.dtype = torch.float16, quant_config: Optional[quark.torch.quantization.config.config.Config] = None, export_type: Optional[str] = None) -> None

      This function aims to export json and safetensors files of the quantized Pytorch model.
      The model's network architecture or configuration is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the safetensors file.
      :param model: The quantized model to be exported.
      :type model: torch.nn.Module
      :param model_type: The type of the model, e.g. gpt2, gptj, llama or gptnext.
      :type model_type: str
      :param model_dtype: The weight data type of the quantized model. Default is torch.float16.
      :type model_dtype: torch.dtype
      :param quant_config: Configuration object containing settings for quantization. Default is None.
      :type quant_config: Optional[Config]
      :param export_type: The specific format in which the JSON and safetensors files are stored. Default is None.
                          The file list of the default exporting format is the same as the original HuggingFace file list. On the basis of these files, add quantization information into them.
                          If set to 'vllm-adopt', the exported files are customized for the VLLM compiler. This option is going to be deprecated soon.
      :type export_type: Optional[str]

      :returns: None

      **Examples**:

          .. code-block:: python

              # default exporting:
              export_path = "./output_dir"
              from quark.torch import ModelExporter
              from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
              exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
              exporter.export_model_info(model, quant_config=quant_config)

          .. code-block:: python

              # vllm adopted exporting:
              export_path = "./output_dir"
              from quark.torch import ModelExporter
              from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
              exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
              exporter.export_model_info(model, model_type=model_type, model_dtype=model_dtype, export_type="vllm-adopt")

      .. note::

         Currently, default exporting format supports large language models(LLM) in HuggingFace.
         If set to 'vllm-adopt', supported quantization types include fp8, int4_per_group, and w4a8_per_group, and supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.


   .. py:method:: export_onnx_model(model: torch.nn.Module, input_args: Union[torch.Tensor, Tuple[float]], input_names: List[str] = [], output_names: List[str] = [], verbose: bool = False, opset_version: Optional[str] = None, do_constant_folding: bool = True, operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX, uint4_int4_flag: bool = False) -> None

      This function aims to export onnx graph of the quantized Pytorch model.

      :param model: The quantized model to be exported.
      :type model: torch.nn.Module
      :param input_args: Example inputs for this quantized model.
      :type input_args: Union[torch.Tensor, Tuple[float]]
      :param input_names: Names to assign to the input nodes of the onnx graph, in order. Default is empty list.
      :type input_names: List[str]
      :param output_names: Names to assign to the output nodes of the onnx graph, in order. Default is empty list.
      :type output_names: List[str]
      :param verbose: Flag to control showing verbose log or no. Default is False
      :type verbose: bool
      :param opset_version: The version of the default (ai.onnx) opset to target.
                            If not set, it will be valued the latest version that is stable for the current version of PyTorch.
      :type opset_version: Optional[str]
      :param do_constant_folding: Apply the constant-folding optimization. Default is False
      :type do_constant_folding: bool
      :param operator_export_type: Export operator type in onnx graph.
                                   The choices include OperatorExportTypes.ONNX, OperatorExportTypes.ONNX_FALLTHROUGH, OperatorExportTypes.ONNX_ATEN and OperatorExportTypes.ONNX_ATEN_FALLBACK.
                                   Default is OperatorExportTypes.ONNX.
      :type operator_export_type: torch.onnx.OperatorExportTypes
      :param uint4_int4_flag: Flag to indicate uint4/int4 quantized model or not. Default is False.
      :type uint4_int4_flag: bool

      :returns: None

      **Examples**:

          .. code-block:: python

              from quark.torch import ModelExporter
              from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
              exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
              exporter.export_onnx_model(model, input_args)

      .. note::

         Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
         In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.


   .. py:method:: export_gguf_model(model: torch.nn.Module, tokenizer_path: Union[str, pathlib.Path], model_type: str) -> None

      This function aims to export gguf file of the quantized Pytorch model.

      :param model: The quantized model to be exported.
      :type model: torch.nn.Module
      :param tokenizer_path: Tokenizer needs to be encoded into gguf model.
                             This argument specifies the directory path of tokenizer which contains tokenizer.json, tokenizer_config.json and/or tokenizer.model
      :type tokenizer_path: Union[str, Path]
      :param model_type: The type of the model, e.g. gpt2, gptj, llama or gptnext.
      :type model_type: str

      :returns: None

      **Examples**:

          .. code-block:: python

              from quark.torch import ModelExporter
              from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
              exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
              exporter.export_gguf_model(model, tokenizer_path, model_type)

      .. note::

         Currently, only support asymetric int4 per_group weight-only quantization, and the group_size must be 32.
         Supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.



