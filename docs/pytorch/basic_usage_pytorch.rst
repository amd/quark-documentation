AMD Quark for PyTorch
=====================

.. note::  
  
    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

Quantizing a floating-point model with AMD Quark for PyTorch involves the following high-level steps:

1. Load the original floating-point model.
2. Set the quantization configuration.
3. Define the data loader.
4. Use the AMD Quark API to perform an in-place replacement of the model's modules with quantized modules.
5. (Optional) Export the quantized model to other formats, such as ONNX.

Supported Features
------------------

AMD Quark for PyTorch supports the following key features:

+--------------------+-------------------------------------------------+
| Feature Name       | Feature Value                                   |
+====================+=================================================+
| Data Type          | Float16 / Bfloat16 / Int4 / Uint4 / Int8 /      |
|                    | OCP_FP8_E4M3 / OCP_MXFP8_E4M3 / OCP_MXFP6 /     |
|                    | OCP_MXFP4 / OCP_MXINT8                          |
+--------------------+-------------------------------------------------+
| Quant Mode         | Eager Mode / FX Graph Mode                      |
+--------------------+-------------------------------------------------+
| Quant Strategy     | Static quant / Dynamic quant / Weight only      |
|                    | quant                                           |
+--------------------+-------------------------------------------------+
| Quant Scheme       | Per tensor / Per channel / Per group            |
+--------------------+-------------------------------------------------+
| Symmetric          | Symmetric / Asymmetric                          |
+--------------------+-------------------------------------------------+
| Calibration method | MinMax / Percentile / MSE                       |
+--------------------+-------------------------------------------------+
| Scale Type         | Float32 / Float16                               |
+--------------------+-------------------------------------------------+
| KV-Cache Quant     | FP8 KV-Cache Quant                              |
+--------------------+-------------------------------------------------+
| In-Place Replace   | nn.Linear / nn.Conv2d / nn.ConvTranspose2d /    |
| OP                 | nn.Embedding / nn.EmbeddingBag                  |
+--------------------+-------------------------------------------------+
| Pre-Quant          | SmoothQuant                                     |
| Optimization       |                                                 |
+--------------------+-------------------------------------------------+
| Quant Algorithm    | AWQ / GPTQ                                      |
+--------------------+-------------------------------------------------+
| Export Format      | ONNX / Json-Safetensors / GGUF(Q4_1)            |
+--------------------+-------------------------------------------------+
| Operating Systems  | Linux (ROCm/CUDA) / Windows (CPU)               |
+--------------------+-------------------------------------------------+

Basic Example
-------------

This example shows a basic use case on how to quantize the ``opt-125m`` model with the ``int8`` data type for ``symmetric`` ``per tensor`` ``weight-only`` quantization.

.. code-block:: python

   # 1. Set model
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
   model.eval()
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

   # 2. Set quantization configuration
   from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
   from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
   from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
   DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                           qscheme=QSchemeType.per_tensor,
                                           observer_cls=PerTensorMinMaxObserver,
                                           symmetric=True,
                                           scale_type=ScaleType.float,
                                           round_method=RoundType.half_even,
                                           is_dynamic=False)

   DEFAULT_W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
   quant_config = Config(global_quant_config=DEFAULT_W_INT8_PER_TENSOR_CONFIG)

   # 3. Define calibration dataloader (still need this step for weight only and dynamic quantization)
   from torch.utils.data import DataLoader
   text = "Hello, how are you?"
   tokenized_outputs = tokenizer(text, return_tensors="pt")
   calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

   # 4. In-place replacement with quantized modules in model
   from quark.torch import ModelQuantizer
   quantizer = ModelQuantizer(quant_config)
   quant_model = quantizer.quantize_model(model, calib_dataloader)

   # # 5. (Optional) Export onnx
   # # If you want to export the quantized model, please freeze the quantized model first
   # freezed_quantized_model = quantizer.freeze(quant_model)
   # from quark.torch import ModelExporter
   # # Get dummy input
   # for data in calib_dataloader:
   #     input_args = data
   #     break
   # quant_model = quant_model.to('cuda')
   # input_args = input_args.to('cuda')
   # exporter = ModelExporter('export_path')
   # exporter.export_onnx_model(quant_model, input_args)

If the code runs successfully, the terminal displays [QUARK-INFO]:
Model quantization has been completed.

For more detailed information, see the section on
:ref:`Advanced AMD Quark Features for PyTorch <advanced-quark-features-pytorch>`.

