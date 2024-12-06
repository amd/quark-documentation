Exporting Quantized Models
==========================

Quark torch not only supports exporting in popular formats requested by
downstream tools, including ONNX, Json-safetensors, and GGUF, but also
supports saving and loading in the torch environment.

Exporting
---------

.. _quark-torch-onnx-exporting:

Onnx Exporting
~~~~~~~~~~~~~~

PyTorch provides a function to export the ONNX graph at this
`link <https://pytorch.org/docs/stable/onnx_torchscript.html#torch.onnx.export>`__.
Quark supports the export of onnx graph for int4, in8, fp8 , float16 and
bfloat16 quantized models. For int4, int8, and fp8 quantization, the
quantization operators used in onnx graph are
`QuantizerLinear <https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html>`__\ \_\ `DequantizerLinear <https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html>`__
pair. For float16 and bfloat16 quantization, the quantization operators
are the cast_cast pair. Mix quantization of int4/uint4 and int8/uint8 is
not supported currently. In other words, if the model contains both
quantized nodes of uint4/int4 and uint8/int8, this function cannot be
used to export the ONNX graph.m
Only support weight-only and static quantization for now.

Example of Onnx Exporting
*************************

.. code:: python


   export_path = "./output_dir"
   batch_iter = iter(calib_dataloader)
   input_args = next(batch_iter)
   if args.quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
       uint4_int4_flag = True
   else:
       uint4_int4_flag = False

   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   export_config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)

.. _quark-torch-json-safetensors-exporting:

Json-Safetensors Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~~

Json-safetensors exporting format is the default exporting format for
Quark, and the file list of this exporting format is the same as the
file list of the original HuggingFace model, with quantization
information added to these files. Taking the llama2-7b model as an
example, the exported file list and added information are as below:

+------------------------------+--------------------------------------------------------------------------+
| File name                    | Additional Quantization Information                                      |
+------------------------------+--------------------------------------------------------------------------+
| config.json                  | Quantization configurations                                              |
+------------------------------+--------------------------------------------------------------------------+
| generation_config.json       | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| model*.safetensors           | Quantization info (tensors of scaling factor, zero point)                |
+------------------------------+--------------------------------------------------------------------------+
| model.safetensors.index.json | Mapping information of scaling factor and zero point to Safetensors files|
+------------------------------+--------------------------------------------------------------------------+
| special_tokens_map.json      | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| tokenizer_config.json        | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+
| tokenizer.json               | \-                                                                       |
+------------------------------+--------------------------------------------------------------------------+

Example of Json-Safetensors Exporting
*************************************

.. code:: python

   export_path = "./output_dir"
   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
   NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                              pack_method="reorder")
   export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   exporter.export_model_info(model, quant_config=quant_config)

By default, ``ModelExporter.export_model_info`` exports models using a Quark-specific format for the checkpoint and ``quantization_config`` format in the ``config.json`` file. This format may not directly be usable by some downstream libraries (AutoAWQ, vLLM).

Until downstream libraries support Quark quantized models, one may export models so that the weight checkpoint and ``config.json`` file targets a specific downstream libraries, using ``custom_mode="awq"`` or ``custom_mode="fp8"``. Example:

.. code:: python
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   # `custom_mode="awq"` would e.g. use `qzeros` instead of `weight_zero_point`, `qweight` instead of `weight` in the checkpoint.
   # Moreover, the `quantization_config` in the `config.json` file is custom, and the full quark `Config` is not serialized.
   exporter.export_model_info(model, quant_config=quant_config, custom_mode="awq")

Json-Safetensors Importing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark provides the importing function for Json-safetensors export files.
<<<<<<< HEAD
In other words, these files can be reloaded into Quark. After reloading, 
=======
In other words, these files can be reloaded into Quark. After reloading,
>>>>>>> 273cbb9a1e (Update 0.6.0 documentation as per tech writer review)
the weights of the quantized operators in the model are stored in the real_quantized format.

Currently, this importing function supports weight-only, static, and dynamic quantization for
FP8 and AWQ. For other quantization methods, only weight-only and static
quantization are supported.

<<<<<<< HEAD
Example of Json-Safetensors Importing 
=======
Example of Json-Safetensors Importing
>>>>>>> 273cbb9a1e (Update 0.6.0 documentation as per tech writer review)
*************************************

.. code:: python

   from quark.torch import import_model_info
   safetensors_model_dir = "./output_dir/json-safetensors"
   model = import_model_info(model, model_info_dir=safetensors_model_dir)

.. _quark-torch-gguf-exporting:

GGUF Exporting
~~~~~~~~~~~~~~

Currently, only support asymetric int4 per_group weight-only
quantization, and the group_size must be 32.The models supported include
Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.

Example of GGUF Exporting
*************************

.. code:: python

   export_path = "./output_dir"
   from quark.torch import ModelExporter
   from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
   export_config = ExporterConfig(json_export_config=JsonExporterConfig())
   exporter = ModelExporter(config=export_config, export_dir=export_path)
   exporter.export_gguf_model(model, tokenizer_path, model_type)

After running the code above successfully, there will be a ``.gguf``
file under export_path, ``./output_dir/llama.gguf`` for example.

Saving & Loading
----------------

Saving
~~~~~~

Save the network architecture or configurations and parameters of the quantized model.

Support both eager and fx-graph model quantization.

For eager mode quantization, the model's configurations are stored in json file,
and parameters including weight, bias, scale, and zero_point are stored in safetensors file.

For fx_graph mode quantization, the model's network architecture and parameters are stored in pth file.

Example of Saving in Eager Mode
*******************************

.. code:: python

   from quark.torch import save_params
   save_params(model, model_type=model_type, export_dir="./save_dir")

Example of Saving in FX-graph Mode
**********************************

.. code:: python

   from quark.torch.export.api import save_params
   save_params(model,
               model_type=model_type,
               args=example_inputs,
               export_dir="./save_dir",
               quant_mode=QuantizationMode.fx_graph_mode)

Loading
~~~~~~~

Instantiate a quantized model from saved model files, which is generated
using the above saving function.

Support both eager and fx-graph model quantization.

Only support weight-only and static quantization for now.

Example of Loading in Eager Mode
********************************

.. code:: python

   from quark.torch import load_params
   model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

Example of Loading in FX-graph Mode
***********************************

.. code:: python

   from quark.torch.quantization.api import load_params
   model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)

.. toctree::
   :hidden:
   :maxdepth: 1

   export/gguf.rst
