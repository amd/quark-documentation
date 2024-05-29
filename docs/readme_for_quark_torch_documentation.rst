Quark for Pytorch
=================

|python| |PyTorch|

**Quark for PyTorch** aims to provide developers with a flexible,
efficient, and easy-to-use toolkit for quantizing deep learning models
from PyTorch. The current quantization method is based on PyTorch
in-place operator replacement. In particular, the tool provides the key
features and verified models as below:

-  Support symmetric/asymmetric post-training quantization (PTQ)
   strategies (weight-only/dynamic/ static PTQ), for various
   quantization levels (per tensor/channel/group) with multiple data
   types (float16/bfloat16/int4/uint4/int8/fp8 (e4m3fn)).
-  Support **configuring calibration methods**, including MinMax,
   Percentile, and MSE.
-  Support **kv-cache** quantization for large language models.
-  Support advanced quantization algorithms, including **SmoothQuant**,
   **AWQ**, and **GPTQ**, for uint4 quantization on GPU.
-  Support exporting quantized models to **ONNX** and **vLLM-adopted**
   JSON-safetensors format.
-  Support operation on Linux (GPU mode) and Windows (CPU mode)
   operating systems.
-  Provide examples for LLM models, such as OPT, **Llama 2**, **Llama
   3**, **Qwen1.5** models with typical datasets, such as “pileval”,
   “wikitext”. Users can set up their own models and add custom
   datasets.

New Features (Version 0.1.0):
-----------------------------

-  **Pytorch Quantizer Enhancements**:
-  Eager mode support.
-  Post Training Quantization (PTQ) is now available.
-  Automatic in-place replacement of ``nn.module`` operations.
-  Quantization of the following modules supported: ``torch.nn.linear``.
-  Customizable calibration process introduced.
-  **Quantization Types**:
-  Symmetric and asymmetric quantization are supported.
-  Weight-only, dynamic, and static quantization modes available.
-  **Quantization Granularity**:
-  Support for per-tensor, per-channel, and per-group granularity.
-  **Data Types**:
-  Multiple data types supported, including float16, bfloat16, int4,
   uint4, int8, and fp8 (e4m3fn).
-  **Calibration Methods**:
-  MinMax, Percentile, and MSE calibration methods now supported.
-  **Large Model Support**:
-  FP8 KV-cache quantization for large language models(LLMs).
-  **Advanced Quantization Algorithms**:
-  Support SmoothQuant, AWQ(uint4), and GPTQ(uint4) for LLMs.
-  **Note**: AWQ/GPTQ/SmoothQuant algorithms are currently limited to
   single GPU usage.
-  **Export Capabilities**:
-  Export of Q/DQ quantized models to ONNX and vLLM-adopted
   JSON-safetensors format now supported.
-  **Operating System Support**:
-  Linux (supports ROCM and CUDA)
-  Windows (support CPU only). Known Issues: AWQ/GPTQ/SmoothQuant
   algorithms are currently limited to single GPU usage.

.. |python| image:: https://img.shields.io/badge/python-3.9%2B-green
   :target: https://www.python.org/
.. |PyTorch| image:: https://img.shields.io/badge/PyTorch-2.2%2B-green
   :target: https://pytorch.org/
