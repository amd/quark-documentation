Accessing ONNX Examples
=======================

Users can get the example code after downloading and unzipping ``amd_quark.zip`` (referring to :doc:`Installation Guide <../install>`).
The example folder is in amd_quark.zip.

   Directory Structure of the ZIP File:

   ::

         + amd_quark.zip
            + amd_quark.whl
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

ONNX Examples in AMD Quark for This Release
-------------------------------------------

.. toctree::
   :caption: Improving Model Accuracy
   :maxdepth: 1

   Block Floating Point (BFP) <example_quark_onnx_BFP>
   MX Formats <example_quark_onnx_MX>
   Fast Finetune AdaRound <example_quark_onnx_adaround>
   Fast Finetune AdaQuant <example_quark_onnx_adaquant>
   Cross-Layer Equalization (CLE) <example_quark_onnx_cle>
   GPTQ <example_quark_onnx_gptq>
   Mixed Precision <example_quark_onnx_mixed_precision>
   Smooth Quant <example_quark_onnx_smoothquant>
   QuaRot <example_quark_onnx_quarot>

.. toctree::
   :caption: Dynamic Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model <example_quark_onnx_dynamic_quantization_llama2>
   Quantizing an OPT-125M Model <example_quark_onnx_dynamic_quantization_opt>

.. toctree::
   :caption: Image Classification
   :maxdepth: 1

   Quantizing a ResNet50-v1-12 Model <example_quark_onnx_image_classification>

.. toctree::
   :caption: Language Models
   :maxdepth: 1

   Quantizing an OPT-125M Model <example_quark_onnx_language_models>

.. toctree::
   :caption: Weights-Only Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model Using the ONNX MatMulNBits <example_quark_onnx_weights_only_quant_int4_matmul_nbits_llama2>
   Quantizating Llama-2-7b model using MatMulNBits <example_quark_onnx_weights_only_quant_int8_qdq_llama2>

.. toctree::
   :caption: Ryzen AI Quantization
   :maxdepth: 1

   Best Practice for Quantizing an Image Classification Model <image_classification_example_quark_onnx_ryzen_ai_best_practice>
   Best Practice for Quantizing an Object Detection Model  <object_detection_example_quark_onnx_ryzen_ai_best_practice>
