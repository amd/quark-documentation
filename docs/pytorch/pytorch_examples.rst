Accessing PyTorch Examples
===========================

You can get the example code after downloading and unzipping ``quark_amd.zip`` (refer to :doc:`Installation Guide <../install>`).
The example folder is in quark_amd.zip.

   Directory Structure of the ZIP File:

   ::

         + quark.zip
            + quark_amd.whl
            + examples    # HERE ARE THE EXAMPLES
               + torch    # HERE ARE THE PYTORCH EXAMPLES
                  + language_modeling
                  + diffusers
                  + ...
               + onnx
                  + image_classification
                  + language_models
                  + ...
            + ...

PyTorch Examples in Quark for This Release
-----------------------------------------

- :doc:`Diffusion Model Quantization <example_quark_torch_diffusers>`
- :doc:`Quark Extension for Brevitas Integration <example_quark_torch_brevitas>`
- :doc:`Integration with AMD Pytorch-light (APL) <example_quark_torch_pytorch_light>`
- :doc:`Language Model Pruning <example_quark_torch_llm_pruning>`
- :doc:`Language Model PTQ <example_quark_torch_llm_ptq>`
- :doc:`Language Model QAT <example_quark_torch_llm_qat>`
- :doc:`Language Model Evaluation <example_quark_torch_llm_eval>`
- :doc:`Vision Model Quantization using FX Graph Mode <example_quark_torch_vision>`
