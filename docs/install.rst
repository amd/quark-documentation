Installation Guide
==================

Install from TAR
----------------

1. Install `PyTorch <https://pytorch.org/>`_ for the compute platform(CUDA, ROCM, CPU...). Version of torch >= 2.2.0.

2. Download the `📥quark.tar.gz <ADD%20TAR%20LINK%20HERE!!!>`__. Extract
   the downloaded tar file and there is a whl package in it.

3. Install quark whl package by

   ::

      pip install [quark whl package].whl

4. (Optional) Verify the installation by running
   ``python -c "import quark"``. If it does not report error, the
   installation is done.

5. (Optional) Compile the ``fast quantization kernels``. When using
   Quark’s quantization APIs for the first time, it will compile the
   ``fast quantization kernels`` using your installed Torch and CUDA if
   available. This process may take a few minutes but subsequent
   quantization calls will be much faster. To invoke this compilation
   now and check if it is successful, run the following command:

   .. code:: bash

      python -c "import quark.torch.kernel"
