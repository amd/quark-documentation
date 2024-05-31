User Guide
==========

There are several steps to quantize a floating-point model with
``Quark for PyTorch``:

1. Load original float model
2. Set quantization configuration
3. Define dataloader
4. Use the Quark API to perform in-place replacement of the model’s
   modules with quantized module.
5. (Optional) Export quantized model to other format such as ONNX

`Quick Start Example <./user_guide_quick_start.md>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Configuring Quark for PyTorch <./user_guide_config_description.md>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Adding Calibration Datasets <./user_guide_dataloader.md>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Exporting for ONNX & Json-Safetensors(vLLM Adopted) <./user_guide_exporting.md>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Feature Description <./user_guide_feature_description.md>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

..
  ------------

  #####################################
  License
  #####################################

  Quark is licensed under MIT License. Refer to the LICENSE file for the full license text and copyright notice.