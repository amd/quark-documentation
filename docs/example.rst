Examples
========

.. contents::
    :local:

How to get the examples
-----------------------

Users can get the example code after downloading and unzipping ``quark.zip`` (referring to `Installation Guide <install.html>`__). The example folder is in quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + quark.whl
            + examples    # HERE IS THE EXAMPLES
               + torch
                  + language_modeling
                  + diffusers
                  + ...
               + onnx
                  + image_classification
                  + language_models
                  + ... 
            + ...

Examples of Quark for Pytorch
-----------------------------

* `Language Model Quantization & Export <./quark_example_torch_llm_gen.html>`__
* `Diffusion Model Quantization & Export <./quark_example_torch_diffusers_gen.html>`__
* `Vision Model Quantization using Quark FX Graph Mode <./quark_example_torch_vision_gen.html>`__
* `Extension for Pytorch-light (AMD internal project) <./quark_example_torch_pytorch_light_gen.html>`__
* `Extension for Brevitas <./quark_example_torch_brevitas_gen.html>`__


Examples of Quark for ONNX
--------------------------
   
* `Image Classification Quantization <./quark_example_onnx_image_classification_gen.html>`__
* `Dynamic Quantization <../../examples/onnx/dynamic_quantization/README.html>`__
* `Fast Finetune AdaRound <./quark_examples_onnx_adaround_gen.html>`__
* `Fast Finetune AdaQuant <./quark_example_onnx_adaquant_gen.html>`__
* `BFP Quantization <./quark_example_onnx_BFP_gen.html>`__
* `Mixed Precision <./quark_onnx_example_mixed_precision_gen.html>`__
* `Cross-Layer Equalization <./quark_example_onnx_cle_gen.html>`__

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
