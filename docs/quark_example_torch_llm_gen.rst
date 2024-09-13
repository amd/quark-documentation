Language Model Quantization Using Quark
=======================================

This document provides examples of quantizing and exporting the language models(OPT, Llama…) using Quark.


Table of Contents
=================

.. contents::
  :local:
  :depth: 1

Supported Models
----------------

+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| Model Name                              | FP8①| INT②  | MX③  | AWQ/GPTQ(INT)④  | SmoothQuant | Rotation |
+=========================================+=====+=======+======+=================+=============+==========+
| meta-llama/Llama-2-*-hf ⑤               | ✓   | ✓     | ✓    | ✓               | ✓           | ✓        |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| meta-llama/Llama-3-*-hf                 | ✓   | ✓     | ✓    | ✓               | ✓           | ✓        |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| meta-llama/Llama-3.1-*-hf               | ✓   | ✓     | ✓    | ✓               | ✓           | ✓        |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| facebook/opt-*                          | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| EleutherAI/gpt-j-6b                     | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| THUDM/chatglm3-6b                       | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| Qwen/Qwen-*                             | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| Qwen/Qwen1.5-*                          | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| Qwen/Qwen1.5-MoE-A2.7B                  | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| Qwen/Qwen2-*                            | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| microsoft/phi-2                         | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| microsoft/Phi-3-mini-*k-instruct        | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| microsoft/Phi-3.5-mini-instruct         | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| mistralai/Mistral-7B-v0.1               | ✓   | ✓     | ✓    | ✓               | ✓           |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| mistralai/Mixtral-8x7B-v0.1             | ✓   | ✓     |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| hpcai-tech/grok-1                       | ✓   | ✓     |      | ✓               |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| CohereForAI/c4ai-command-r-plus-08-2024 | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| CohereForAI/c4ai-command-r-08-2024      | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| CohereForAI/c4ai-command-r-plus         | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| CohereForAI/c4ai-command-r-v01          | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| databricks/dbrx-instruct                | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+
| deepseek-ai/deepseek-moe-16b-chat       | ✓   |       |      |                 |             |          |
+-----------------------------------------+-----+-------+------+-----------------+-------------+----------+


.. note::
   - ① FP8 means ``OCP fp8_e4m3`` data type quantization.
   - ② INT includes INT8, UINT8, INT4, UINT4 data type quantization
   - ③ MX includes OCP data type MXINT8, MXFP8E4M3, MXFP8E5M2, MXFP4, MXFP6E3M2, MXFP6E2M3.
   - ④ GPTQ only supports QuantScheme as 'PerGroup' and 'PerChannel'.
   - ⑤ ``*`` represents different model sizes, such as ``7b``.

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta.
For additional details, see the `Llama2 page on Huggingface <https://huggingface.co/docs/transformers/main/en/model_doc/llama2>`__. Upon obtaining permission, download the checkpoint to the ``[llama2_checkpoint_folder]``.

Quantization & Export Scripts
-----------------------------

You can run the following python scripts in the ``examples/torch/language_modeling`` path. Here we use Llama2-7b as an example.

Note:

1. To avoid memory limitations, GPU users can add the ``--multi_gpu`` argument when running the model on multiple GPUs.
2. CPU users should add the ``--device cpu`` argument.

**Recipe 1: Evaluation of Llama2 float16 model without quantization**

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --skip_quantization

**Recipe 2: FP8(OCP fp8_e4m3) Quantization & Json_SafeTensors_Export with KV cache**

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_fp8_a_fp8 \
                             --kv_cache_dtype fp8 \
                             --num_calib_data 128 \
                             --model_export quark_safetensors


**Recipe 3: INT Wight Only Quantization & Json_SafeTensors_Export of Llama2 with AWQ**

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int4_per_group_sym \
                             --num_calib_data 128 \
                             --quant_algo awq \
                             --dataset pileval_for_awq_benchmark \
                             --seq_len 512 \
                             --model_export quark_safetensors


**Recipe 4: INT Static Quantization & Json_SafeTensors_Export of Llama2 with AWQ (on CPU)**

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int8_a_int8_per_tensor_sym \
                             --num_calib_data 128 \
                             --device cpu \
                             --model_export quark_safetensors


**Recipe 5: Quantization & GGUF_Export with AWQ (W_uint4 A_float16 per_group asymmetric)**

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_uint4_per_group_asym \
                             --quant_algo awq \
                             --num_calib_data 128 \
                             --group_size 32 \
                             --model_export gguf

If the code runs successfully, it will produce one gguf file in ``[output_dir]`` and the terminal will display ``GGUF quantized model exported to ... successfully.``


**Recipe 6: MX Quantization**

Quark now supports the datatype microscaling which is abbreviated as MX. Use the following command to quantize model to datatype MX:

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mx_fp8 \
                             --num_calib_data 32 \
                             --group_size 32

The command above is weight-only quantization. If users want activations to be quantized as well, use the command below:

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mx_fp8_a_mx_fp8 \
                             --num_calib_data 32 \
                             --group_size 32


**Recipe 7: BFP16 Quantization**

Quark now supports the datatype BFP16 which is short for block floating point 16 bits. Use the following command to quantize model to datatype BFP16:

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_bfp16 \
                             --num_calib_data 16

The command above is weight-only quantization. If users want activations to be quantized as well, use the command below:

::

   python3 quantize_quark.py --model_dir [llama2 checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_bfp16_a_bfp16 \
                             --num_calib_data 16


Tutorial: Running a Model Not on the Supported List
---------------------------------------------------

For a new model that is not listed in Quark, you need to modify some relevant files.
There are several steps to follow.

-  Step 1: add the model type to ``MODEL_NAME_PATTERN_MAP`` in ``get_model_type`` function in quantize_quark.py.
-  Step 2: customize ``tokenizer`` for your model in ``get_tokenizer`` function in quantize_quark.py.
-  Step 3: [Optional] for some layers you don't want to quantize, add them to ``MODEL_NAME_EXCLUDE_LAYERS_MAP`` in configuration_preparation.py.
-  Step 4: [Optional] if quantizing ``kv_cache``, you must add name of kv layers to ``MODEL_NAME_KV_LAYERS_MAP`` in configuration_preparation.py.
-  Step 5: [Optional] if using GPTQ, SmoothQuant and AWQ, add ``awq_config.json`` and ``gptq_config.json`` for model.


Step 1: Add the model type to ``MODEL_NAME_PATTERN_MAP`` in ``get_model_type`` function in quantize_quark.py.
_____________________________________________________________________________________________________________
``MODEL_NAME_PATTERN_MAP`` describes ``model type``, which is used to configure the quant_config for the models.
You can use the part of the model's HF-ID as the key of the dictionary, and the lowercase version of this key as the value.
For ``CohereForAI/c4ai-command-r-v01``, you can add ``{"Cohere": "cohere"}`` to ``MODEL_NAME_PATTERN_MAP``.

.. code:: python

    def get_model_type(model: nn.Module) -> str:
        MODEL_NAME_PATTERN_MAP = {
            "Llama": "llama",
            "OPT": "opt",
            ...
            "Cohere": "cohere",  # <---- Add code HERE
        }
        for k, v in MODEL_NAME_PATTERN_MAP.items():
            if k.lower() in type(model).__name__.lower():
                return v

Step 2: Customize ``tokenizer`` for your model in ``get_tokenizer`` function in quantize_quark.py.
__________________________________________________________________________________________________
For the most part, ``get_tokenizer`` function is applicable. But for some models, such as ``CohereForAI/c4ai-command-r-v01``, ``use_fast`` can only be set to ``True`` (as of ``transformers-4.44.2``).
You can customize the ``tokenizer`` by referring to your model's ``Model card`` on ``Hugging Face`` and tokenization_auto.py in ``transformers``.

.. code:: python

    def get_tokenizer(ckpt_path: str, max_seq_len: int = 2048, model_type: Optional[str] = None) -> AutoTokenizer:
        print(f"Initializing tokenizer from {ckpt_path}")
        use_fast = True if model_type == "grok" or model_type == "cohere" else False
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                                model_max_length=max_seq_len,
                                                padding_side="left",
                                                trust_remote_code=True,
                                                use_fast=use_fast)

Step 3: [Optional] For some layers you don't want to quantize, add them to ``MODEL_NAME_EXCLUDE_LAYERS_MAP`` in configuration_preparation.py.
_____________________________________________________________________________________________________________________________________________
Normally, if you are quantizing a MoE model, the ``gate`` layers do not need to be quantized, or there are other layers that you do not want to quantize, you can add ``model_type`` and ``excluding layer name`` to ``MODEL_NAME_EXCLUDE_LAYERS_MAP``.
You can add the name of the layer or part of the name with wildcards.
For ``dbrx-instruct``, you can add ``"dbrx": ["lm_head", "*router.layer"]`` to ``MODEL_NAME_EXCLUDE_LAYERS_MAP``.
Note that ``lm_head`` is excluded by default.

.. code:: python

    MODEL_NAME_EXCLUDE_LAYERS_MAP = {
            "llama": ["lm_head"],
            "opt": ["lm_head"],
            ...
            "cohere": ["lm_head"],  # <---- Add code HERE
            }

Step 4: [Optional] If quantizing ``kv_cache``, you must add name of kv layers to ``MODEL_NAME_KV_LAYERS_MAP`` in configuration_preparation.py.
______________________________________________________________________________________________________________________________________________

When quantizing ``kv_cache``, you must add ``model_type`` and ``kv layers name`` to ``MODEL_NAME_KV_LAYERS_MAP``.
For ``facebook/opt-125m``, the full name of ``k_proj`` is ``model.model.decoder.layer[0].self_attn.k_proj`` (similar for ``v_proj``),
add the names with wildcards like ``"opt": ["*k_proj", "*v_proj"]``.
For ``chatglm``, you can add ``"chatglm": ["*query_key_value"]``.

.. code:: python

    MODEL_NAME_KV_LAYERS_MAP = {
            "llama": ["*k_proj", "*v_proj"],
            "opt": ["*k_proj", "*v_proj"],
            ...
            "cohere": ["*k_proj", "*v_proj"],  # <---- Add code HERE
            }


Step 5: [Optional] If using GPTQ, SmoothQuant and AWQ, add ``awq_config.json`` and ``gptq_config.json`` for model.
__________________________________________________________________________________________________________________

Quark relies on ``awq_config.json`` and ``gptq_config.json`` to execute GPTQ, SmoothQuant and AWQ.
Therefore, you must create a model directory named after the ``model_type`` mentioned in Step1 under ``Quark/examples/torch/language_modeling/models`` and create ``awq_config.json`` and ``gptq_config.json`` in this directory.
Take the ``meta-llama/Llama-2-7b`` model as an example, we create directory named ``llama`` in ``Quark/examples/torch/language_modeling/models``,
and create ``awq_config.json`` and ``gptq_config.json`` in ``Quark/examples/torch/language_modeling/models/llama``.

For GPTQ
++++++++

The config file should be named by ``gptq_config.json``. You should collate all linear layers in decoder layers, and put them in ``inside_layer_modules`` list,
and put the decoder layers name in ``model_decoder_layers`` list.
You can refer to ``Quark/examples/torch/language_modeling/models/*/gptq_config.json``, and find the configuration of a model with a similar structure to your model.

For SmoothQuant and AWQ
+++++++++++++++++++++++

SmoothQuant and AWQ use same file named ``awq_config.json``.
In general, for each decoder layer, you need to process four parts (linear_qkv, linear_o, linear_mlp_fc1, linear_mlp_fc2).
You should provide them with the previous adjacent layer (``prev_op``), input layer (``inp``), inspecting layer (``module2inspect``).
If there is a necessary condition to inspect, you can use ``condition`` to check, ``help`` is optional and can provide additional information.
Additionally, when you quantize a model with GQA, ``num_attention_heads`` and ``num_key_value_heads`` should be added to ``awq_config.json``, and ``alpha`` should be specified specifically as ``0.85``, which influences how aggressively weights are quantized.
At last, put the decoder layers name in ``model_decoder_layers``.
You can refer to ``Quark/examples/torch/language_modeling/models/*/awq_config.json``, and find the configuration of a model with a similar structure to your model.
For example, models containing the GPA structure can refer to ``Quark/examples/torch/language_modeling/models/qwen2moe/awq_config.json``,
and those containing the moe structure can refer to ``Quark/examples/torch/language_modeling/models/grok/awq_config.json``.

Tutorial: Generating AWQ Configuration Automatically (Experimental)
-------------------------------------------------------------------

We provide a script `awq_auto_config_helper.py` to simplify user operations by quickly identifying modules compatible with the "AWQ" and "SmoothQuant" algorithms within the model through torch.compile.

Installation
____________

This script requires PyTorch version 2.4 or higher.

Usage
_____

The `MODEL_DIR` variable should be set to the model name from Hugging Face, such as `facebook/opt-125m`, `Qwen/Qwen2-0.5B`, or `EleutherAI/gpt-j-6b`.

To run the script, use the following command::

    MODEL_DIR="your_model"
    python awq_auto_config_helper.py --model_dir "${MODEL_DIR}"

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
