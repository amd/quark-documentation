Quark
=====

.. image:: https://img.shields.io/badge/license-MIT-blue
   :target: LICENSE
.. image:: https://img.shields.io/badge/python-3.9%2B-green
   :target: https://www.python.org/


**Quark** is a comprehensive cross-platform toolkit designed to simplify and enhance the quantization of deep learning models. Supporting both PyTorch and ONNX models, Quark empowers developers to optimize their models for deployment on a wide range of hardware backends, achieving significant performance gains without compromising accuracy.

Quark for PyTorch: Flexible and Efficient Quantization for PyTorch Models
-------------------------------------------------------------------------

Quark for PyTorch provides developers with a flexible, efficient, and easy-to-use toolkit for quantizing deep learning
models from PyTorch. The current quantization method is based on PyTorch in-place operator replacement.
In particular, the tool provides the key features and verified models as below:

Key Features
^^^^^^^^^^^^

* **Comprehensive Quantization Support**:
   - **Eager Mode Post-Training Quantization (PTQ):** Quantize pre-trained models without the need for retraining data.
   - **FX Graph Mode PTQ and Quantization-Aware Training (QAT):** Optimize models during training for superior accuracy on quantized hardware.
   - **Optimized QAT Methods:** Support Trained Quantization Thresholds For Accurate And Efficient Fixed-Point Inference Of Deep Neural Networks (TQT), Learned Step Size Quantization (LSQ) for better QAT result.
   - **Flexible Quantization Strategies:** Choose from symmetric/asymmetric, weight-only/static/dynamic quantization, and various quantization levels (per tensor/channel) to fine-tune performance and accuracy trade-offs.
   - **Extensive Data Type Support:** Quantize models using a wide range of data types, including `float16`, `bfloat16`, `int4`, `uint4`, `int8`, `fp8 (e4m3fn and e5m2)`, Shared Microexponents with Multi-Level Scaling (`MX6`, `MX9``), and `Microscaling (MX)` data types with `int8`, `fp8_e4m3fn`, `fp8_e5m2`, `fp4`, `fp6_e3m2`, and `fp6_e2m3` elements.
   - **Configurable Calibration Methods:** Optimize quantization accuracy with `MinMax`, `Percentile`, and `MSE` calibration methods.
* **Advanced Capabilities:**
   - **Large Language Model Optimization:** Specialized support for quantizing large language models with `kv-cache` quantization.
   - **Cutting-Edge Algorithms:** Leverage state-of-the-art algorithms like `SmoothQuant`, `AWQ`, and `GPTQ` for `uint4` quantization on GPUs, achieving optimal performance for demanding tasks.
* **Seamless Integration and Deployment:**
   - **Export to multiple formats:** Export quantized models to `ONNX`, `JSON-safetensors`, and `GGUF` formats for deployment on a wide range of platforms.
   - **APL Integration:** Seamlessly integrate with AMD Pytorch-light (APL) for optimized performance on AMD hardware, to provide `INT-K`, `BFP16`, and `BRECQ` support.
   - **Experimental Brevitas Integration:** Explore seamless integration with Brevitas for quantizing Stable Diffusion and ImageNet classification models.
* **Examples included:** Benefit from practical examples for LLM models, SDXL models (Eager Mode), and CNN models (FX Graph Mode), accelerating your quantization journey.
* **Cross-Platform Support:** Develop and deploy on both Linux (CPU and GPU) and Windows (CPU mode) operating systems.

Quark for ONNX: Streamlined Quantization for ONNX models
--------------------------------------------------------

Quark for ONNX leverages the power of the ONNX Runtime Quantization tool,
providing a robust and flexible solution for quantizing ONNX models.

Key Features
^^^^^^^^^^^^

* **Comprehensive Quantization Support**:
   - **Post-Training Quantization (PTQ):** Quantize pre-trained models without the need for retraining data.
   - **Flexible Quantization Strategies:** Choose from symmetric/asymmetric, weight-only/static/dynamic quantization, and various quantization levels (per tensor/channel) to fine-tune performance and accuracy trade-offs.
   - **Extensive Data Type Support:** Quantize models using a wide range of data types, including `uint32`, `int32`, `float16`, `bfloat16`, `int16`, `uint16`, `int8`, `uint8` and `bfp`.
   - **Configurable Calibration Methods:** Optimize quantization accuracy with `MinMax`, `Entropy`, `Percentile`, `NonOverflow` and `MinMSE` calibration methods.
* **Advanced Capabilities:**
   - **Multiple Deployment Targets:** Target a variety of hardware platforms, including `NPU_CNN`, `NPU_Transformer`, and `CPU`.
   - **Cutting-Edge Algorithms:** Leverage state-of-the-art algorithms like `SmoothQuant`, `CLE`, `BiasCorrection`, `AdaQuant`, and `AdaRound`, achieving optimal performance for demanding tasks.
   - **Flexible Scale Types:** Support quantization with `float scale`, `int16 scale`, and `power-of-two scale` options.
   - **Automatic Mixed Precision:**  Achieve an optimal balance between accuracy and performance through automatic mixed precision.

For further details on the features and capabilities of Quark, please refer to the
`📖Documentation <https://quark.docs.amd.com>`__ and
`💡Examples <./example.html>`__ pages.

Installation
------------

Binaries
^^^^^^^^

Commands to install binaries via pip wheels or zip files can be found on our
`🛠️Installation Guide <./install.html>`__

From Source
^^^^^^^^^^^

To install Quark from source for either Windows or Linux, follow the steps below:

**Get Quark Source Code:**

.. code:: bash

   git clone --recursive https://gitenterprise.com/AMDNeuralOpt/Quark
   cd Quark
   # if you are updating an existing checkout
   git submodule sync
   git submodule update --init --recursive

**Install Prerequisites:**

If you are installing from source, you will need:

   * Python 3.9 or later
   * Install PyTorch >= 2.2.0
   * Install ONNX >= 1.12.0
   * Install ONNX Runtime >= 1.17.0, <1.19.0
   * Install ONNX Runtime Extensions >= 0.4.2

We highly recommend installing an `Anaconda <https://docs.anaconda.com/anaconda/install/>`__  environment.

The `requirements.txt` file contains the necessary dependencies listed for Quark. To install these dependencies, run:

.. code:: bash

   pip install -r requirements.txt

By default, the `requirements.txt` file **does not** contain the *PyTorch* package because it depends on your
Operating System and acceleration hardware (e.g. CPU, CUDA, ROCm, etc).
Follow the steps from the `PyTorch <https://pytorch.org/get-started/locally/>`__ website to install the
appropriate PyTorch package for your system.

**Build and Install Quark:**

Now that you have the prerequisites installed, you can build and install Quark by running:

.. code:: bash

   pip install .

For more information, including installation verification steps,
please refer to the `🛠️Installation Guide <./install.html>`__.

Releases and contributing
-------------------------

Quark is in very active development with several releases a year.
Please let us know if you encounter a bug by `filing an issue (internal only) <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/issues>`__.

Any contribution is much appreciated, and the following are some recommendations:

* If you are planning to contribute bug-fixes, please do so without any further discussion.
* If you plan to contribute new features, or extensions to the core, please open an issue and discuss the feature with us first.

To learn more about making a contribution to Quark,
please see our `Contributing (internal only) <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/CONTRIBUTING.md>`__ page.
For more information about Quark releases, see `Releases (internal only) <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/releases>`__ page.

Communication
-------------

* GitHub Issues: Bug reports, feature requests, install issues, RFCs, and any feedback, etc.

The Team
--------

Quark is an AMD project led by `Spandan Tiwari <spandan.tiwari@amd.com>`__ and is maintained by several skillful
engineers and researchers contributing to it.
Refer to `CODEOWNERS (internal only) <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/CODEOWNERS>`__ to identify
the team members responsible for each part of the project.

Resources
---------

- `📖Documentation <https://quark.docs.amd.com>`__: Contains **Getting Started**, **APIs**, **User Guide**, and other detailed information.
- `💡Examples <./example.html>`__: Examples of Language Model and Image Classification are provided to demonstrate the usage of Quark.
- `📄FAQ <./faq.html>`__: Check out our FAQ for more details.

License
-------

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
