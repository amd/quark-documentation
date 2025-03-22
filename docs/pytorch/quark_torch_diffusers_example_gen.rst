SDXL Model Quantization using Quark
===================================

This document provides examples of FP8 quantizing and exporting the SDXL
models using Quark.

Third-party Dependencies
------------------------

The example relies on ``torchvision``, User need to install the version
of ``torchvision`` that is compatible with their version of PyTorch.

.. code:: shell

   export DIFFUSERS_ROOT=$PWD
   git clone https://github.com/mlcommons/inference.git
   cd inference
   git checkout 87ba8cb8a6a4f6525f26255fa513d902b17ab060
   cd ./text_to_image/tools/
   sh ./download-coco-2014.sh --num-workers 5
   sh ./download-coco-2014-calibration.sh -n 5
   cd ${DIFFUSERS_ROOT}
   export PYTHONPATH="${DIFFUSERS_ROOT}/inference/text_to_image/:$PYTHONPATH"

Dataset Files
-------------

-  The calibration dataset file will be downloaded to
   ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/calibration/captions.tsv``.
-  The test dataset file will be downloaded to
   ``${DIFFUSERS_ROOT}/inference/text_to_image/coco2014/captions/captions_source.tsv``.

Quantization & Export Scripts
-----------------------------

You can run the following python scripts in the
``examples/torch/diffusers`` path.

Run with SDXL Without Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Run original SDXL: 

--------------------------------------

.. code::

   python quantize_sdxl.py --float


Calibration and Export SafeTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
-  Run Calibration:

--------------------------------------

.. code::

   python quantize_sdxl.py --input_scheme {'per-tensor'} --weight_scheme {'per-tensor', 'per-channel'} --calib_data_tsv_file_path {your calibration dataset file path} --export

Load SafeTensor and Test
~~~~~~~~~~~~~~~~~~~~~~~~

-  Load and Test:

--------------------------------------

.. code::

   python quantize_sdxl.py --input_scheme {'per-tensor'} --weight_scheme {'per-tensor', 'per-channel'}  --test_data_tsv_file_path {your calibration dataset file path} --load --test

Load SafeTensor and Run with a prompt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Load and Run:

--------------------------------------

.. code::

   python quantize_sdxl.py --input_scheme {'per-tensor'} --weight_scheme {'per-tensor', 'per-channel'} --load --prompt "A city at night with people walking around."

SDXL Benchmark
--------------

**MI210** GPU, diffusers==0.21.2

+----------------+-------------------------------+------------------+
| Model Name     | FP16  (Without Quantization)  | FP8 + Per-Tensor |
+================+===============================+==================+
| clip score     | 31.74845                      | 31.83954         |
+----------------+-------------------------------+------------------+
| fid            | 23.56758                      | 23.614748        |
+----------------+-------------------------------+------------------+