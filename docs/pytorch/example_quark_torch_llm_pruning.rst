Pruning
=======

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <../pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_pruning``.

This topic contains examples of pruning language models (such as OPT and Llama) using Quark.

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. Access the Llama2 models checkpoint by submitting a permission request to Meta. For additional details, see the Llama2 page on Huggingface. Upon obtaining permission, download the checkpoint to the ``[llama2_checkpoint_folder]``.

Pruning Scripts
---------------

Run the following Python scripts in the ``examples/torch/language_modeling/llm_pruning`` path. Use Llama2-7b as an example.

.. note::

    - To avoid memory limitations, GPU users can add the ``--multi_gpu`` argument when running the model on multiple GPUs.
    - CPU users should add the ``--device cpu`` argument.

Recipe 1: Evaluation of Llama2 Float16 Model without Pruning
------------------------------------------------------------

.. code-block:: bash

    python3 main.py --model_dir [llama2 checkpoint folder] \
                             --skip_pruning

Recipe 2: Pruning Model and Saved to Safetensors
------------------------------------------------

.. code-block:: bash

    python3 main.py --model_dir [llama2 checkpoint folder] \
                             --pruning_algo "osscar" \
                             --num_calib_data 128 \
                             --save_pruned_model \
                             --save_dir save_dir
