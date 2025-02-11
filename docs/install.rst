Installation Guide
==================

Prerequisites
-------------

1. Python 3.9, 3.10, or 3.11 is required.
    * Python 3.12 is currently **unsupported**.
2. Install `PyTorch <https://pytorch.org/>`__ for your compute platform (such as CUDA, ROCM, and CPU).
    * **Only select** PyTorch version 2.2.0 or later.
3. Install the following:
    * `ONNX <https://onnx.ai/>`__ version 1.16.0 or later,
    * `ONNX Runtime <https://onnxruntime.ai/>`__ version 1.17.0 or later, **but earlier than or equal to 1.20.1**,
    * `onnxruntime-extensions <https://onnxruntime.ai/docs/extensions/>`__ version 0.4.2 or later.

.. note::
   When installing on Windows, Visual Studio is necessary, with Visual Studio 2022 being the minimum required version. During the compilation process, you can choose one of the following methods:

   1. Use the Developer Command Prompt for Visual Studio. When installing Visual Studio, ensure to include the Developer Command Prompt. You can execute the programs in the CMD window of the Developer Command Prompt for Visual Studio.
   2. Manually add paths to the environment variables. The tools `cl.exe`, `MSBuild.exe`, and `link.exe` from Visual Studio are utilized. Ensure that their paths are included in the PATH environment variable. These programs can be found in the Visual Studio installation directory. In the Edit Environment Variables window, click **New**, and then paste the path to the folder containing the `cl.exe`, `link.exe`, and `MSBuild.exe` files. Click **OK** on all the windows to apply the changes.

**Note**: AMD Quark package has been renamed to amd_quark. Please use the new package name for releases newer than 0.6.0.

Installation
------------

Install from ZIP
~~~~~~~~~~~~~~~~

**Step 1**: Download and unzip 📥*amd_quark-\*.zip* which has a wheel package in it. You can download wheel package 📥*amd_quark-\*.whl* directly.

   `📥amd_quark.zip release_version (recommend) <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-@version@.zip>`__

   `📥amd_quark.whl release_version <https://www.xilinx.com/bin/public/openDownload?filename=amd_quark-@version@-py3-none-any.whl>`__

   Directory Structure of the zip file:

   ::

      + amd_quark.zip
         + amd_quark.whl
         + examples    # Examples code of Quark
         + docs        # Off-line documentation of Quark.
         + README.md

   We strongly recommend downloading the zip file, as it includes examples compatible with the wheel package version.

**Step 2**: Install the quark wheel package by running the following command:

   ::

      pip install amd_quark*.whl

Installation Verification
-------------------------

1. (Optional) Verify the installation by running
   ``python -c "import quark"``. If no error is reported, the installation is successful.

2. (Optional) Compile the ``fast quantization kernels``.
   When using Quark's quantization APIs for the first time, it compiles the ``fast quantization kernels`` using your installed Torch and CUDA, if available.
   This process might take a few minutes, but the subsequent quantization calls are much faster.
   To invoke this compilation now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.torch.kernel"

3. (Optional) Compile the ``custom operators library``.
   When using Quark-ONNX's custom operators for the first time, it compiles the ``custom operators library`` using your local environment.
   To invoke this compilation now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.onnx.operators.custom_ops"

Older Versions
--------------

**Note**: The following links are for older versions of AMD Quark, before the package has been renamed to amd_quark.

-  `quark_0.7.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.7.zip>`__
-  `quark_0.6.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.6.0.zip>`__
-  `quark_0.5.1.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.1+88e60b456.zip>`__
-  `quark_0.5.1.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.1+88e60b456.zip>`__
-  `quark_0.5.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.5.0+fae64a406.zip>`__
-  `quark_0.2.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23.zip>`__
-  `quark_0.1.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5.zip>`__

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
