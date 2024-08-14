Tutorial: Mixed Precision
=========================

Introduction
============

In this tutorial, we will learn how to use Mixed Precision in Quark.

What is Mixed Precision Quantization?
-------------------------------------

Mixed precision quantization involves using different precision levels
for different parts of a neural network, such as using 8-bit integers
for some layers while retaining higher precision (e.g., 16-bit or 32-bit
floating point) for others. This approach leverages the fact that not
all parts of a model are equally sensitive to quantization. By carefully
selecting which parts of the model can tolerate lower precision, mixed
precision quantization achieves significant computational savings while
minimizing the impact on model accuracy.

Key Concepts
~~~~~~~~~~~~

1. **Layer-wise Precision Assignment**: Different layers of a neural
   network can have varying levels of sensitivity to quantization. Mixed
   precision quantization assigns precision levels to layers based on
   their sensitivity, optimizing both performance and accuracy.
2. **Loss Sensitivity Analysis**: A crucial step in mixed precision
   quantization is determining how sensitive each layer is to precision
   reduction. This can be done through techniques like sensitivity
   analysis, which measures the impact of quantization on the loss
   function.
3. **Hybrid Precision Representation**: By combining multiple precision
   formats (e.g., FP32, FP16, INT8) within a single model, mixed
   precision quantization maximizes computational efficiency while
   maintaining high accuracy where it is most needed.
4. **Auto Mixed Precision**: Quark for ONNX supports the auto mixed
   precision that automatically determine the precision levels for each
   node based on the acceptable accuracy loss defined by user.

Benefits of Mixed Precision Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Enhanced Efficiency**: By using lower precision where possible,
   mixed precision quantization significantly reduces computational load
   and memory usage, leading to faster inference times and lower power
   consumption.
2. **Maintained Accuracy**: By selectively applying higher precision to
   sensitive parts of the model, mixed precision quantization minimizes
   the accuracy loss that typically accompanies uniform quantization.
3. **Flexibility**: Mixed precision quantization is adaptable to various
   types of neural networks and can be tailored to specific hardware
   capabilities, making it suitable for a wide range of applications.

How to enable Mixed Precision in Quark for ONNX?
------------------------------------------------

Please refer to this
`link <./user_guide_accuracy_improvement.html#1.2-quantizing-using-mix-precision>`__
for more details about how to enable Mixed Precision in configuration of
Quark for ONNX.

Layer Wise Mixed Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~

Here is a simple example of how to enable layer wise mixed precision in
Quark for ONNX.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig
   from typing import Dict

   quant_config = QuantizationConfig(
       calibrate_method=quark.onnx.PowerOfTwoMethod.NonOverflow,
       quant_format=quark.onnx.VitisQuantFormat.QDQ,
       activation_type=quark.onnx.VitisQuantType.QInt16,
       weight_type=quark.onnx.QuantType.QInt8,
       specific_tensor_precision=True,
       # MixedPrecisionTensor is a dictionary in which the key is data type like int8/int16, 
       # and the value is a list of the names of tensors to be quantized to that data type.
       extra_options={"MixedPrecisionTensor":{data_type_0:['tensor_0', 'tensor_1', 'tensor_2'], 
                                              data_type_1:['tensor_0', 'tensor_1', 'tensor_2']}}
   )
   config = Config(global_quant_config=quant_config)

Auto Mixed Precision
~~~~~~~~~~~~~~~~~~~~

Here is a simple example of how to enable auto mixed precision in Quark
for ONNX.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       calibrate_method=quark.onnx.CalibrationMethod.Percentile,
       quant_format=quark.onnx.VitisQuantFormat.QDQ,
       activation_type=quark.onnx.VitisQuantType.QInt16,
       weight_type=QuantType.QInt8,
       include_auto_mp=True,
       extra_options={
           'AutoMixprecision': {
               'ActTargetQuantType':QuantType.QInt8,
               'WeightTargetQuantType'::QuantType.QInt8,
               'OutputIndex': 0,
               'Top1AccTarget': 0.1,
               'EvaluateFunction': top1_acc,
           },
       },
   )
   config = Config(global_quant_config=quant_config)

Examples
--------

Here is an example of quantizing a densenet121.ra_in1k model using the
mixed precision provided in Quark for ONNX in
examples/onnx/accuracy_improvement/mixed_precision/README.
