Installation Guide
==================

Prerequisites
-------------

1. Python 3.9+ is required.
2. Install `PyTorch <https://pytorch.org/>`__ for the compute
   platform(CUDA, ROCM, CPU…). Version of torch >= 2.2.0.
3. Install `ONNX <https://onnx.ai/>`__ of version >= 1.12.0, `ONNX
   Runtime <https://onnxruntime.ai/>`__ of version ~= 1.17.0,
   `onnxruntime-extensions <https://onnxruntime.ai/docs/extensions/>`__
   of version >= 0.4.2

Installation
------------

Install from ZIP
~~~~~~~~~~~~~~~~

1. Download the
   `📥quark.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23.zip>`__.
   Extract the downloaded zip file and there is a whl package in it. Or You can download whl package directly.
   `📥quark.whl <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.2.0+6af1bac23-py3-none-any.whl>`__.

2. Install quark whl package by

   .. code:: bash

      pip install [quark whl package].whl

Installation Verification
-------------------------

1. (Optional) Verify the installation by running
   ``python -c "import quark"``. If it does not report error, the
   installation is done.

2. (Optional) Compile the ``fast quantization kernels``. When using
   Quark’s quantization APIs for the first time, it will compile the
   ``fast quantization kernels`` using your installed Torch and CUDA if
   available. This process may take a few minutes but subsequent
   quantization calls will be much faster. To invoke this compilation
   now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.torch.kernel"

3. (Optional) Compile the ``custom operators library``. When using
   Quark-ONNX’s custom operators for the first time, it will compile the
   ``custom operators library`` using your local environment. To invoke
   this compilation now and check if it is successful, run the following
   command:

   .. code:: bash

      python -c "import quark.onnx.operators.custom_ops"

Old version zip
---------------

-  `quark_0.1.0.zip <https://www.xilinx.com/bin/public/openDownload?filename=quark-0.1.0+a9827f5.zip>`__.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
