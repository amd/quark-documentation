Language Model Evaluation in Quark
==================================

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <../pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_eval``.

This topic provides examples of evaluating large language models using the Quark evaluation API. Evaluate models either by calculating perplexity (PPL) on WikiText2 or `benchmark tasks <https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks>`_. The evaluated models can be either pre-trained or quantized, and evaluation can be performed after the model is exported from Quark or during the quantization pipeline. Use the PTQ method for evaluation as an example. For QAT or pruning, employ the evaluation APIs in a similar way.

Quick Start
-----------

.. code-block:: bash

    pip install -r requirements.txt

Note that perplexity evaluation is performed by default in the PTQ pipeline. For task evaluation, add new tasks to the evaluation list as follows.

Evaluate Pre-trained Model without Quantization: PPL and openllm
----------------------------------------------------------------

.. code-block:: bash

    cd ../llm_ptq
    python3 quantize_quark.py --model_dir [model checkpoint] \
                              --skip_quantization \
                              --tasks openllm

Evaluate Quantized Model during PTQ Pipeline: PPL and MMLU
----------------------------------------------------------

.. code-block:: bash

    python3 quantize_quark.py --model_dir [model checkpoint] \
                              --output_dir output_dir \
                              --quant_scheme w_fp8_a_fp8 \
                              --kv_cache_dtype fp8 \
                              --num_calib_data 128 \
                              --tasks mmlu \
                              --eval_batch_size 8 \
                              --model_export quark_safetensors

Evaluate Quantized Model Exported by Quark: PPL and arc_challenge, hellaswag
----------------------------------------------------------------------------

.. code-block:: bash

    python3 quantize_quark.py --model_dir [model checkpoint] \
                              --safetensors_model_reload \
                              --safetensors_model_dir [output_dir] \
                              --tasks arc_challenge,hellaswag
