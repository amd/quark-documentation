Installation Guide
==================

Prerequisites
-------------

1. Python 3.9+ is required.
2. Install `PyTorch <https://pytorch.org/>`__ for the compute platform(CUDA, ROCM, CPU…). Version of torch >= 2.2.0.
3. Install `ONNX <https://onnx.ai/>`__ of version >= 1.16.0, `ONNX Runtime <https://onnxruntime.ai/>`__ of version >= 1.17.0, <1.20.0
.0,
   `onnxruntime-extensions <https://onnxruntime.ai/docs/extensions/>`__ of version >= 0.4.2
4. If you are running in an environment that already has a transformers version below 4.44.0, please update it to version 4.44.0 or higher.

**Note**: When installing on Windows, Visual Studio is required. The minimum version of Visual Studio is Visual Studio 2022. During the compilation process,There are two ways to use it:

1. Use the Developer Command Prompt for Visual Studio, When installing Visual Studio, ensure that Developer Command Prompt for Visual Studio is installed as well. Execute programs in the CMD window of Developer Command Prompt for Visual Studio.
2. Manually Add Paths to Environment Variables, Visual Studio's cl.exe, MSBuild.exe and link.exe will be used. Please ensure that the paths are added to the PATH environment variable. Those programs are located in the Visual Studio installation directory. In the Edit Environment Variables window, click New, then paste the path to the folder containing cl.exe, link.exe and MSBuild.exe. Click OK on all the windows to apply the changes.

Installation
------------

Install from ZIP
~~~~~~~~~~~~~~~~

**Step 1**: Download and unzip 📥quark.zip and there is a whl package in it. Or you can download whl package 📥quark.whl directly.

   `📥quark.zip release_version (recommend) <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.0+fae64a406.zip>`__

   `📥quark.whl release_version <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.0+fae64a406-py3-none-any.whl>`__

   `📥quark.zip nightly_version (internal only) <https://xcoartifactory/ui/native/uai-pip-local/com/amd/quark/main/nightly/>`__

   Directory Structure of zip file:

   ::

      + quark.zip
         + quark.whl
         + examples    # Examples code of Quark
         + docs        # Off-line documentation of Quark.
         + README.md

   We strongly recommend users download the zip file as it includes examples compatible with the whl package version.

**Step 2**: Install quark whl package by

   ::

      pip install [quark whl package].whl

Install from source code (internal only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark is an internal project and the source code is hosted on Xilinx Github Enterprise.
To install Quark from source, follow the steps below:

1. Download and open repository:

   .. code:: bash

      git clone https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark.git
      cd Quark

2. Install other dependencies:

   .. code:: bash

      pip install -r requirements.txt

3. Install the main component:

   .. code:: bash

      pip install -e .

Installation Verification
-------------------------

1. (Optional) Verify the installation by running
   ``python -c "import quark"``. If it does not report error, the installation is done.

2. (Optional) Compile the ``fast quantization kernels``. 
   When using Quark's quantization APIs for the first time, it will compile the ``fast quantization kernels`` using your installed Torch and CUDA if available. 
   This process may take a few minutes but subsequent quantization calls will be much faster. 
   To invoke this compilation now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.torch.kernel"

3. (Optional) Compile the ``custom operators library``. 
   When using Quark-ONNX's custom operators for the first time, it will compile the ``custom operators library`` using your local environment. 
   To invoke this compilation now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.onnx.operators.custom_ops"

Old version zip
---------------

-  `quark_0.2.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23.zip>`__
-  `quark_0.2.0.whl <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23-py3-none-any.whl>`__
-  `quark_0.1.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5.zip>`__

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
