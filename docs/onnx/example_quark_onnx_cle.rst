.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing a resnet152 model using
the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization without CLE <#quantization-without-cle>`__
-  `Quantization with CLE <#quantization-with-cle>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r ../utils/requirements.txt

Prepare model
-------------

Export onnx model from resnet152 torch model. The corresponding model link is https://huggingface.co/timm/resnet152.a1h_in1k.:

::

   mkdir models && python ../utils/export_onnx.py resnet152

Prepare data
------------

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access
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

Quantization without CLE
------------------------

The quantizer takes the float model and produce a quantized model
without CLE.

::

   python quantize_model.py --model_name resnet152 \
                            --input_model_path models/resnet152.onnx \
                            --output_model_path models/resnet152_quantized.onnx \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models**
folder, which was quantized by S8S8_AAWS configuration (Int8 symmetric
quantization) without CLE.

Quantization with CLE
---------------------

The quantizer takes the float model and produce a quantized model with
CLE.

::

   python quantize_model.py --model_name resnet152 \
                            --input_model_path models/resnet152.onnx \
                            --output_model_path models/resnet152_cle_quantized.onnx \
                            --include_cle \
                            --calibration_dataset_path calib_data

This command will generate a quantized model under the **models**
folder, which was quantized by S8S8_AAWS configuration (Int8 symmetric
quantization) with CLE.

Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152.onnx

Test the accuracy of the quantized model without CLE on ImageNet val
dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152_quantized.onnx

Test the accuracy of the quantized model with CLE on ImageNet val
dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152_cle_quantized.onnx

+-------+--------------------+---------------------+------------------+
|       | Float Model        | Quantized Model     | Quantized Model  |
|       |                    | without CLE         | with CLE         |
+=======+====================+=====================+==================+
| Model | 232 MB             | 59 MB               | 59 MB            |
| Size  |                    |                     |                  |
+-------+--------------------+---------------------+------------------+
| P     | 83.456 %           | 70.042 %            | 79.664 %         |
| rec@1 |                    |                     |                  |
+-------+--------------------+---------------------+------------------+
| P     | 96.580 %           | 88.502 %            | 94.854 %         |
| rec@5 |                    |                     |                  |
+-------+--------------------+---------------------+------------------+

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
