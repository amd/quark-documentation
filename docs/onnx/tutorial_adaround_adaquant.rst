Tutorial: AdaRound and AdaQuant
===============================

In this tutorial, we will learn how to use AdaRound and AdaQuantin in
Quark for ONNX.

Introduction
------------

AdaRound
~~~~~~~~

**AdaRound**, short for “Adaptive Rounding,” is a post-training
quantization technique that aims to minimize the accuracy drop typically
associated with quantization. Unlike standard rounding methods, which
can be too rigid and cause significant deviations from the original
model’s behavior, Adaround uses an adaptive approach to determine the
optimal rounding of weights. Here is the
`link <https://arxiv.org/abs/2004.10568>`__ to the paper.

AdaQuant
~~~~~~~~

**AdaQuant**, short for “Adaptive Quantization,” is an advanced
quantization technique designed to minimize the accuracy loss typically
associated with post-training quantization. Unlike traditional static
quantization methods, which apply uniform quantization across all layers
and weights, AdaQuant dynamically adapts the quantization parameters
based on the characteristics of the model and its data. Here is the
`link <https://arxiv.org/abs/1712.01048>`__ to the paper.

Benefits of Adaround and AdaQuant
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Improved Accuracy**: By minimizing the quantization error, Adaround
   helps preserve the model’s accuracy closer to its original state. By
   dynamically adjusting quantization parameters, AdaQuant helps retain
   a higher level of model accuracy compared to traditional quantization
   methods.
2. **Flexibility**: Adaround and AdaQuant can be applied to various
   layers and types of neural networks, making it a versatile tool for
   different quantization needs.
3. **Post-Training Application**: Adaround does not require retraining
   the model from scratch. It can be applied after the model has been
   trained, making it a convenient choice for deploying pre-trained
   models in resource-constrained environments.
4. **Efficiency**: AdaQuant enables the deployment of high-performance
   models in resource-constrained environments, such as mobile and edge
   devices, without the need for extensive retraining.

Upgrades of AdaRound / AdaQuant in Quark for ONNX
-------------------------------------------------

Comparing with the original algorithm, AdaRound in Quark for ONNX is
modified and upgraded to be more flexible.

1. **Unified Framework**: These two algorithms were integrated into a
   unified framework named as “fast finetune”.
2. **Quantization Aware Finetuning**: Only the weight and bias
   (optional) will be updated, the scales and zero points are fixed,
   which ensures that all the quantizing informations and the structure
   of the quantized model keep unchanged after finetuning.
3. **Flexibility**: AdaRound in Quark for ONNX is compatible with many
   more graph patterns-matching.
4. **More Advanced Options**

   -  **Early Stop**: If average loss of the current batch iterations
      decreases comparing with the previous batch of iterations, the
      training of the layer will stop early. It will accelerate the
      finetuning process.
   -  **Selective Update**: If the end-to-end accuracy does not improve
      after trained a certain layer, discard the finetuning result of
      that layer.
   -  **Adjust Learning Rate**: Besides the overall learning rate, users
      could set up a scheme to adjust learning rate layer wise. For
      example, apply a larger learning rate on the layer that has a
      bigger loss.

How to enable AdaRound / AdaQuant in Quark?
-------------------------------------------

AdaRound and AdaQuant are provided as options of optimal algorithms for
fast finetune. Please refer to this
`link <./user_guide_accuracy_improvement.rst#1.3-quantizing-using-fast-finetune>`__
for more details about how to set the configuration to enable AdaRound
or AdaQuant.

Here is an simple example showing how to enable default AdaRound and
AdaQuant configuration.

.. code:: python

   from quark.onnx.quantization.config.config import Config, QuantizationConfig, get_default_config
   # Config of default AdaRound
   quant_config = get_default_config("S8S8_AAWS_ADAROUND")
   config = Config(global_quant_config=quant_config)
   # Config of default AdaQuant
   quant_config = get_default_config("S8S8_AAWS_ADAQUANT")
   config = Config(global_quant_config=quant_config)

Examples
--------

.. _adaround-1:

AdaRound
~~~~~~~~

An example of quantizing a mobilenetv2_050.lamb_in1k model using the
AadRound in ONNX for Quark is provided
examples/onnx/accuracy_improvement/adaround/README. The table below
shows the accuracy improved by applying AdaRound. 

+-------+-------------------+---------------------+-------------------+
|       | Float Model       | Quantized Model     | Quantized Model   |
|       |                   | without ADAROUND    | with ADAROUND     |
+=======+===================+=====================+===================+
| Model | 8.4 MB            | 2.3 MB              | 2.4 MB            |
| Size  |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 65.424 %          | 1.708 %             | 41.420 %          |
| rec@1 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 85.788 %          | 5.690 %             | 64.802 %          |
| rec@5 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+

.. _adaquant-1:

AdaQuant
~~~~~~~~

An example of quantizing a mobilenetv2_050.lamb_in1k model using the
AdaQuant in ONNX for Quark is provided
examples/onnx/accuracy_improvement/adaquant/README. The table below
shows the accuracy improved by applying AdaQuant. 

+-------+-------------------+---------------------+-------------------+
|       | Float Model       | Quantized Model     | Quantized Model   |
|       |                   | without ADAQUANT    | with ADAQUANT     |
+=======+===================+=====================+===================+
| Model | 8.4 MB            | 2.3 MB              | 2.4 MB            |
| Size  |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 65.424 %          | 1.708 %             | 52.322 %          |
| rec@1 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+
| P     | 85.788 %          | 5.690 %             | 75.756 %          |
| rec@5 |                   |                     |                   |
+-------+-------------------+---------------------+-------------------+

References
----------

1. AdaRound

   `Nagel et al., 2020 <https://arxiv.org/abs/2004.10568>`__ Nagel, M.,
   van Baalen, M., Blankevoort, T., & Welling, M. (2020). *Up or Down?
   Adaptive Rounding for Post-Training Quantization*. arXiv:2004.10568.

2. AdaQuant

   `Jacob et al., 2017 <https://arxiv.org/abs/1712.01048>`__ Jacob, B.,
   Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., … & Adam, H.
   (2017). *Quantization and Training of Neural Networks for Efficient
   Integer-Arithmetic-Only Inference*. arXiv:1712.01048.
