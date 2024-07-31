Quark
=====

Installation
------------

1. Install `torch <https://pytorch.org/>`__ and torchvision for the
   compute platform(CUDA, ROCM, CPU…). Version of torch >= 2.2.0.

2. Install quark whl package in current path by

   ::

      pip install [quark whl package].whl

3. (Optional) Verify the installation by running
   ``python -c "import quark"``. If it does not report error, the
   installation is done.

4. (Optional) Compile the ``fast quantization kernels``. When using
   Quark-PyTorch’s quantization APIs for the first time, it will compile
   the ``fast quantization kernels`` using your installed Torch and CUDA
   if available. This process may take a few minutes but subsequent
   quantization calls will be much faster. To invoke this compilation
   now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.torch.kernel"

5. (Optional) Compile the ``custom operators library``. When using
   Quark-ONNX’s custom operators for the first time, it will compile the
   ``custom operators library`` using your local environment. To invoke
   this compilation now and check if it is successful, run the following
   command:

   .. code:: bash

      python -c "import quark.onnx.operators.custom_ops"

Documentation
-------------

For more information about Quark, please refers to the HTML
documentation at ``docs/html/index.html``.

Examples
--------

For examples of large language model quantization, please refer to
``examples/torch/language_modeling/README.md``. For examples of image
classification model quantization, please refer to
``examples/onnx/image_classification/README.md``.

License
-------

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
