.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example for MX Formats
===============================

This folder contains an example of quantizing a ResNet50 model using the ONNX quantizer
of Quark with Microscaling (MX) formats.

Same as Block Floating Point (BFP), the elements in the MX block also share a common exponent, but
they have independent data type, such as FP8 (E5M2 and E4M3), FP6 (E3M2 and E2M3), FP4 (E2M1) and INT8,
which bring about a fine-grained scaling within the block to improve the precision.

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization with MX Formats <#quantization-with-mx-formats>`__
-  `Evaluation <#evaluation>`__


Pip requirements
----------------
Install the necessary python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare model
-------------

Download the onnx float model from the `onnx/models <https://github.com/onnx/models>`__ repo directly:

::

   wget -P models https://github.com/onnx/models/raw/new-models/vision/classification/resnet/model/resnet50-v1-12.onnx


Prepare data
------------

ILSVRC 2012, commonly known as ‘ImageNet’. This dataset provides access
to ImageNet (ILSVRC) 2012 which is the most commonly used subset of
ImageNet. This dataset spans 1000 object classes and contains 50,000
validation images.

If you already have an ImageNet datasets, you can directly use your
dataset path.

To prepare the test data, please check the download section of the main
website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You
need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

::

   mkdir val_data && tar -xzf val_images.tar.gz -C val_data
   python ../utils/prepare_data.py val_data calib_data

The storage format of the val_data of the ImageNet dataset organized as
follows:

-  val_data

   -  n01440764

      -  ILSVRC2012_val_00000293.JPEG
      -  ILSVRC2012_val_00002138.JPEG
      -  …

   -  n01443537

      -  ILSVRC2012_val_00000236.JPEG
      -  ILSVRC2012_val_00000262.JPEG
      -  …

   -  …
   -  n15075141

      -  ILSVRC2012_val_00001079.JPEG
      -  ILSVRC2012_val_00002663.JPEG
      -  …

The storage format of the calib_data of the ImageNet dataset organized
as follows:

-  calib_data

   -  n01440764

      -  ILSVRC2012_val_00000293.JPEG

   -  n01443537

      -  ILSVRC2012_val_00000236.JPEG

   -  …
   -  n15075141

      -  ILSVRC2012_val_00001079.JPEG

Quantization with MX Formats
-----------------------------

The quantizer takes the float model and produces a MX quantized model.
There are built-in configurations in the quantization script for MX formats,
which named as 'MXFP8E5M2', 'MXFP8E4M3', 'MXFP6E3M2', 'MXFP6E2M3', 'MXFP4E2M1', 'MXINT8'.
We can choose different MX formats by passing one of the configuration names to the script.
Here is an example of MXINT8 quantization:

::

   python quantize_model.py --input_model_path models/resnet50-v1-12.onnx \
                            --output_model_path models/resnet50-v1-12_quantized.onnx \
                            --calibration_dataset_path calib_data \
                            --config MXINT8

This command will generate a MX quantized model under the **models** folder.

Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12.onnx

Test the accuracy of the MX quantized model on ImageNet
val dataset:

::

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12_quantized.onnx

If want to run faster with GPU support, you can also execute the following command:


::

   python onnx_validate.py val_data --batch-size 1 --onnx-input models/resnet50-v1-12_quantized.onnx --gpu

+--------+-------------------+---------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
|        | Float Model       | Quantized Model     | Quantized Model   | Quantized Model   | Quantized Model   | Quantized Model   | Quantized Model   |
|        |                   | with MXINT8         | with MXFP8E5M2    | with MXFP8E4M3    | with MXFP6E3M2    | with MXFP6E2M3    | with MXFP4E2M1    |
+========+===================+=====================+===================+===================+===================+===================+===================+
| Model  | 97.82 MB          | 97.47 MB            | 97.47 MB          | 97.47 MB          | 97.47 MB          | 97.47 MB          | 97.47 MB          |
| Size   |                   |                     |                   |                   |                   |                   |                   |
+--------+-------------------+---------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Prec@1 | 74.114 %          | 74.124 %            | 63.388 %          | 69.634 %          | 63.318 %          | 71.612 %          | 4.592 %           |
|        |                   |                     |                   |                   |                   |                   |                   |
+--------+-------------------+---------------------+-------------------+-------------------+-------------------+-------------------+-------------------+
| Prec@5 | 91.716 %          | 91.718 %            | 86.640 %          | 89.630 %          | 86.654 %          | 90.680 %          | 13.450 %          |
|        |                   |                     |                   |                   |                   |                   |                   |
+--------+-------------------+---------------------+-------------------+-------------------+-------------------+-------------------+-------------------+

Note: Different executive devices can lead to minor variations in the
accuracy of quantized model.

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
