Exporting Using ONNX Runtime Gen AI Model Builder
=================================================

This document provides examples of quantizing large language models (LLMs) to **UINT4** using the **AWQ algorithm** via the Quark API, and exporting them to ONNX format using the **ONNX Runtime Gen AI Model Builder**.

**ONNX Runtime Gen AI (OGA)** offers an end-to-end pipeline for working with ONNX models, including inference using ONNX Runtime, logits processing, search and sampling, and key-value (KV) cache management. For detailed documentation, visit the `ONNX Runtime Gen AI Documentation <https://onnxruntime.ai/docs/genai>`_. The tool includes a `Model Builder <https://onnxruntime.ai/docs/genai/howto/build-model.html>`_ that facilitates exporting models to the ONNX format.

Preparation
-----------

Model Preparation
~~~~~~~~~~~~~~~~~

To use **Llama2 models**, download the HF Llama2 checkpoint. Access to these checkpoints requires a permission request to Meta. For more information, refer to the `Llama2 page on Huggingface <https://huggingface.co>`_. Once permission is granted, download the checkpoint and save it to the `<llama checkpoint folder>`.

ONNX Runtime Gen AI (OGA) Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the ONNX Runtime Gen AI package using pip:

.. code-block:: bash

    pip install onnxruntime-genai

Quark UINT4 Quantization with AWQ
---------------------------------

**Quantization Configuration**: AWQ / Group 128 / Asymmetric / FP16 activations

Use the following command to quantize the model:

.. code-block:: bash

    python3 quantize_quark.py --model_dir <llama checkpoint folder> \
                              --output_dir <quantized safetensor output dir> \
                              --quant_scheme w_uint4_per_group_asym \
                              --num_calib_data 128 \
                              --quant_algo awq \
                              --dataset pileval_for_awq_benchmark \
                              --seq_len 512 \
                              --model_export quark_safetensors \
                              --data_type float16 \
                              --custom_mode awq

This will generate a directory containing the safe tensors at the specified `<quantized safetensor output dir>`.

.. note::

    To include the `lm_head` layer in the quantization process, add the `--exclude_layers` flag. This overrides the default behavior of excluding the `lm_head` layer.

.. note::

    To quantize the model for BF16 activations, use the `--data_type bfloat16` flag.

.. note::

    To specify a group size other than 128, such as 32, use the `--group_size 32` flag.

Exporting Using ONNX Runtime Gen AI Model Builder
-------------------------------------------------

To export the quantized model to ONNX format, run the following command:

.. code-block:: bash

    python3 -m onnxruntime_genai.models.builder \
            -i <quantized safetensor output dir> \
            -o <quantized onnx model output dir> \
            -p int4 \
            -e dml

.. note::

    The activation data type of the ONNX model depends on the combination of the `-p` (precision) and `-e` (execution provider) flags. For example:

    - Using `-p int4 -e dml` will generate an ONNX model with float16 activations prepared for the DML execution provider.
    - To generate an ONNX model with float32 activations, use the `-p int4 -e cpu` flag.
