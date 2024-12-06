Accessing ONNX Examples
=======================

Users can get the example code after downloading and unzipping ``quark.zip`` (referring to :doc:`Installation Guide <install>`).
The example folder is in quark.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + quark.whl
            + examples    # HERE IS THE EXAMPLES
               + torch
                  + language_modeling
                  + diffusers
                  + ...
               + onnx # HERE ARE THE ONNX EXAMPLES
                  + image_classification
                  + language_models
                  + ...
            + ...

ONNX Examples in Quark for This Release
---------------------------------------

Improving Model Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`Block Floating Point (BFP) <example_quark_onnx_BFP>`
- :doc:`MX Formats <example_quark_onnx_MX>`
- :doc:`Fast Finetune AdaRound <example_quark_onnx_adaround>`
- :doc:`Fast Finetune AdaQuant <example_quark_onnx_adaquant>`
- :doc:`Cross-Layer Equalization (CLE) <example_quark_onnx_cle>`
- :doc:`GPTQ <example_quark_onnx_gptq>`
- :doc:`Mixed Precision <example_quark_onnx_mixed_precision>`
- :doc:`Smooth Quant <example_quark_onnx_smoothquant>`

Dynamic Quantization
~~~~~~~~~~~~~~~~~~~~

- :doc:`Quantizing an Llama-2-7b Model <example_quark_onnx_dynamic_quantization_llama2>`
- :doc:`Quantizing an OPT-125M Model <example_quark_onnx_dynamic_quantization_opt>`

Image Classification
~~~~~~~~~~~~~~~~~~~~

- :doc:`Quantizing a ResNet50-v1-12 Model <example_quark_onnx_image_classification>`

Language Models
~~~~~~~~~~~~~~~

- :doc:`Quantizing an OPT-125M Model <example_quark_onnx_language_models>`

Weights-Only Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`Quantizing an Llama-2-7b Model Using the ONNX MatMulNBits <example_quark_onnx_weights_only_quant_int4_matmul_nbits_llama2>`
