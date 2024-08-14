Quark for Pytorch - Exporting
=============================

Onnx Exporting
--------------

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

Example of Onnx Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python


   export_path = "./output_dir"
   batch_iter = iter(calib_dataloader)
   input_args = next(batch_iter)
   if args.quant_scheme in ["w_int4_per_channel_sym", "w_uint4_per_group_asym", "w_int4_per_group_sym", "w_uint4_a_bfloat16_per_group_asym"]:
       uint4_int4_flag = True
   else:
       uint4_int4_flag = False

   from quark.torch import ModelExporter
   from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
   exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
   exporter.export_onnx_model(model, input_args, uint4_int4_flag=uint4_int4_flag)

Json-Safetensors Exporting
--------------------------

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


For fp8 per_tensor quantization, this exporting format is the same as
the exporting format of AutoFP8. And for AWQ quantization, this
exporting format is the same as the exporting format of AutoAWQ when the
version is 'gemm'.

Example of Json-Safetensors Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   export_path = "./output_dir"
   from quark.torch import ModelExporter
   from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
   exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
   exporter.export_model_info(model, quant_config=quant_config)

GGUF Exporting
--------------

Currently, only support asymetric int4 per_group weight-only
quantization, and the group_size must be 32.The models supported include
Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.

Example of GGUF Exporting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   export_path = "./output_dir"
   from quark.torch import ModelExporter
   from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
   exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
   exporter.export_gguf_model(model, tokenizer_path, model_type)

After running the code above successfully, there will be a ``.gguf``
file under export_path, ``./output_dir/llama.gguf`` for example.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
