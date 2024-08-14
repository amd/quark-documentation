Quark
=====

.. image:: https://img.shields.io/badge/release-0.2.0-blue
   :target: quark/version.txt
.. image:: https://img.shields.io/badge/license-MIT-blue
   :target: LICENSE
.. image:: https://img.shields.io/badge/python-3.9%2B-green
   :target: https://www.python.org/

**Quark** is a deep learning model quantization toolkit for quantizing models from PyTorch, ONNX, and other frameworks. It provides easy-to-use APIs for quantization and more advanced features than native frameworks, supporting multiple HW backends.

**Quark for PyTorch** provides developers with a flexible, efficient, and easy-to-use toolkit for quantizing deep learning models from PyTorch. The current quantization method is based on PyTorch in-place operator replacement. In particular, the tool provides the key features and verified models as below:

- Support **Eager Mode** post-training quantization (PTQ) based on PyTorch in-place operator replacement.
- Support **FX Graph Mode** PTQ and Quantization-Aware Training (QAT) based on torch.fx.GraphModule.
- Support symmetric/asymmetric quantization strategies (weight-only/dynamic/static quantization), for various quantization levels (per tensor/channel/group).
- Support data types as float16/bfloat16/int4/uint4/int8/fp8 (e4m3fn) and Microscaling (MX) data types as int8, fp8(e4m3fn), fp4, fp6_e3m2, and fp6_e2m3
- Support **configuring calibration methods**, including MinMax, Percentile, and MSE.
- Support **kv-cache** quantization for large language models.
- Support advanced quantization algorithms, including **SmoothQuant**, **AWQ**, and **GPTQ**, for uint4 quantization on GPU.
- Support exporting quantized models to **ONNX**, **JSON-safetensors**, and **GGUF** format.
- Support operation on Linux and Windows (CPU mode) operating systems.
- Provide examples for LLM models and the SDXL model in Eager Mode. Provide CNN models in FX Graph Mode.
- Provide the integrated example with APL(AMD Pytorch-light, internal project name), supporting the invocation of APL's INT-K, BFP16, and BRECQ.
- Provide the experimental Quark extension interface, enabling seamless integration of Brevitas for Stable Diffusion and Imagenet classification model quantization.

**Quark for ONNX** provides developers with a flexible, efficient, and easy-to-use toolkit for quantizing deep learning models from ONNX. The tool is based on `Quantization Tool <https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/quantization>`_ of ONNXRuntime and provides the key features as below:

- Support symmetric/asymmetric post-training quantization (PTQ) strategies (weight-only/static PTQ), for various quantization levels (per tensor/channel) with multiple data types (uint32/int32/float16/bfloat16/int16/uint16/int8/uint8/bfp).
- Support configuring different calibration methods, including MinMax, Entropy, Percentile, NonOverflow, and MinMSE.
- Support multiple deployment targets, including NPU_CNN, NPU_Transformer, and CPU.
- Support advanced quantization algorithms, including **CLE**, **BiasCorrection**, **AdaQuant**, **AdaRound**, and **SmoothQuant**.
- Support various scale types, including float scale, int16 scale, and power-of-two scale.
- Support automatic mixing precision to balance accuracy and performance.
- Support operation on Linux and Windows operating systems.

Installation
------------

- `🛠️Installation Guide <./install.html>`__

Resources
---------

- `📖Documentation <./index.html>`__: Contains **Getting Started**, **APIs**, **User Guide**, and other detailed information.
- `💡Examples <./example.html>`__: Examples of Language Model and Image Classification are provided to demonstrate the usage of Quark.
- `📄FAQ <./faq.html>`__: Check out our FAQ for more details.

License
-------

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
