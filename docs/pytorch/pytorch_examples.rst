Examples
========

How to get the examples
-----------------------

Users can get the example code after downloading and unzipping ``quark.zip`` (referring to :doc:`Installation Guide <install>`).
The example folder is in quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + quark.whl
            + examples    # HERE IS THE EXAMPLES
               + torch
                  + language_modeling
                     + llm_pruning
                     + llm_ptq
                     + llm_qat
                     + llm_eval
                  + vision
                  + diffusers
                  + ...
               + onnx
                  + image_classification
                  + language_models
                  + ... 
            + ...

Examples of Quark for Pytorch
-----------------------------

* :doc:`Language Model Pruning <example_quark_torch_llm_pruning>`
* :doc:`Language Model Post Training Quantization(PTQ) & Export <example_quark_torch_llm_ptq>`
* :doc:`Language Model Quantization Aware Training (QAT) <example_quark_torch_llm_qat>`
* :doc:`Language Model Evaluation <example_quark_torch_llm_eval>`
* :doc:`Diffusion Model Quantization & Export <example_quark_torch_diffusers>`
* :doc:`Vision Model Quantization using Quark FX Graph Mode <example_quark_torch_vision>`
* :doc:`Extension for Pytorch-light (AMD internal project) <example_quark_torch_pytorch_light>`
* :doc:`Extension for Brevitas <example_quark_torch_brevitas>`

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
