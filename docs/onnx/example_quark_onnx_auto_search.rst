.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example
===============================

This folder contains an example of Auto search for quantizing a yolov3 model based on
the ONNX quantizer of Quark. The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Prepare data <#prepare-data>`__
-  `Quantization with auto_search <#quantization-with-auto_search>`__


Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r ./requirements.txt

Prepare model
-------------

Download the yolov3 model from huggingface url:

::

   https://huggingface.co/amd/yolov3/tree/main

Prepare data
------------

COCO 2017, commonly known as 'COCO'. This dataset include five thsound
validation pictures with labels.

In this example, we use the buildin evaluator, so we do not use the
COCO2017 directly. Instead, at first you need to prepare the calibration
dataset with coco 2017 preprocss and save in .npy format.


The storage format of the val_data of the COCO2017 dataset organized as
follows:

-  val_data
      -  sample_1.npy
      -  sample_2.npy
      -  …

we use this dataset as evluation dataset and calibration dataset at the same time.


Quantization with auto_search
------------------------

The quantize config, input model, calibration dataset is default.
so we only need to excute the start script run.sh

::

   python auto_search_model.py --model_name "yolov3" --input_model_path $YOLOV3_FLOAT_ONNX_PATH

This command will generate a series of configs from the auto_search config.
When stop condition is False, the instance will sample config from the whole
search space according the search algorithm.
Then the input model will be quantied using quark onnx and the sampled config.
Based on the metric and the evaluator, the quantized model will calculate the
metirc and vilidate that if it is within the tolerance. If the metric statisfy
the tolerance, the quantized model will be moved to the  output directionary,
otherwise the model will be



Evaluation
----------

Test the accuracy of the float model on ImageNet val dataset:

::

   python ../utils/onnx_validate.py val_data --model-name resnet152 --batch-size 1 --onnx-input models/resnet152.onnx

Test the accuracy of the quantized model without CLE on ImageNet val
dataset:




.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
