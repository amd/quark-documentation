.. raw:: html

   <!-- omit in toc -->

Quark ONNX Quantization Example
===============================

This folder contains an example of quantizing a opt-125m model using the ONNX quantizer of Quark. It also shows how to use the Smooth Quant algorithm.

The example has the following parts:

-  `Pip requirements <#pip-requirements>`__
-  `Prepare model <#prepare-model>`__
-  `Quantization without Smooth_Quant <#quantization-without-smooth-quant>`__
-  `Quantization with Smooth_Quant <#quantization-with-smooth-quant>`__
-  `Evaluation <#evaluation>`__

Pip requirements
----------------

Install the necessary python packages:

::

   python -m pip install -r requirements.txt

Prepare model
-------------

Get opt-125m torch model:

::

   mkdir opt-125m
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/pytorch_model.bin
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/tokenizer_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/vocab.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/merges.txt
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/generation_config.json
   wget -P opt-125m https://huggingface.co/facebook/opt-125m/resolve/main/special_tokens_map.json


Export onnx model from opt-125m torch model:

::

   mkdir models && optimum-cli export onnx --model ./opt-125m --task text-generation ./models/

Quantization without Smooth Quant
---------------------------------

The quantizer takes the float model and produce a quantized model without Smooth Quant.

::

   cp -r models quantized_models && rm quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path quantized_models/quantized_model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT

This command will generate a quantized model under the **quantized_models** folder, which was quantized by Int8 configuration for transformer-based models.

Quantization with Smooth Quant
------------------------------

The quantizer takes the float model and produce a quantized model with Smooth Quant.

::

   cp -r models smoothed_quantized_models && rm smoothed_quantized_models/model.onnx
   python quantize_model.py --input_model_path models/model.onnx \
                            --output_model_path smoothed_quantized_models/smoothed_quantized_model.onnx \
                            --config INT8_TRANSFORMER_DEFAULT \
                            --include_sq

This command will generate a quantized model under the **smoothed_quantized_models** folder, which was quantized by Int8 configuration for transformer-based models with Smooth Quant.

Evaluation
----------

Test the PPL of the float model on wikitext2.raw:

::

   python onnx_validate.py --model_name_or_path models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model models/ --do_onnx_eval --no_cuda

Test the PPL of the quantized model without Smooth Quant:

::

   python onnx_validate.py --model_name_or_path quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model quantized_models/ --do_onnx_eval --no_cuda

Test the PPL of the quantized model with Smooth Quant:

::

   python onnx_validate.py --model_name_or_path smoothed_quantized_models/ --per_gpu_eval_batch_size 1 --block_size 2048 --onnx_model smoothed_quantized_models/ --do_onnx_eval --no_cuda

+-------+--------------------+---------------------+------------------+
|       | Float Model        | Quantized Model     | Quantized Model  |
|       |                    | without Smooth Quant| with Smooth Quant|
+=======+====================+=====================+==================+
| Model | 480 MB             | 384 MB              | 385 MB           |
| Size  |                    |                     |                  |
+-------+--------------------+---------------------+------------------+
| PPL   | 27.0317            | 28.6846             | 28.4315          |
+-------+--------------------+---------------------+------------------+

.. raw:: html

   <!-- omit in toc -->

License
-------

Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
