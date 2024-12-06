Mixed Precision
===============

In this topic, you will learn how to use Mixed Precision in Quark.

What is Mixed Precision Quantization?
-------------------------------------

Mixed precision quantization involves using different precision levels for different parts of a neural network, such as using 8-bit integers for some layers while retaining higher precision, for example, 16-bit or 32-bit floating point, for others. This approach leverages the fact that not all parts of a model are equally sensitive to quantization. By carefully selecting which parts of the model can tolerate lower precision, you achieve significant computational savings while minimizing the impact on model accuracy.

Key Concepts
------------

1. **Layer-wise Precision Assignment**: Different layers of a neural network can have varying levels of sensitivity to quantization. Mixed precision quantization assigns precision levels to layers based on their sensitivity, optimizing both performance and accuracy.

2. **Loss Sensitivity Analysis**: A crucial step in mixed precision quantization is determining how sensitive each layer is to precision reduction. This can be done through techniques like sensitivity analysis, which measures the impact of quantization on the loss function.

3. **Hybrid Precision Representation**: By combining multiple precision formats, for example, FP32, FP16, INT8, within a single model, mixed precision quantization maximizes computational efficiency while maintaining high accuracy where it is most needed.

4. **Auto Mixed Precision**: Quark for ONNX supports auto mixed precision that automatically determines the precision levels for each node based on the acceptable accuracy loss defined by you.

Benefits of Mixed Precision Quantization
----------------------------------------

1. **Enhanced Efficiency**: By using lower precision where possible, mixed precision quantization significantly reduces computational load and memory usage, leading to faster inference times and lower power consumption.

2. **Maintained Accuracy**: By selectively applying higher precision to sensitive parts of the model, mixed precision quantization minimizes the accuracy loss that typically accompanies uniform quantization.

3. **Flexibility**: Mixed precision quantization is adaptable to various types of neural networks and can be tailored to specific hardware capabilities, making it suitable for a wide range of applications.

How to enable Mixed Precision in Quark for ONNX?
------------------------------------------------

For more details about how to enable Mixed Precision in the configuration of Quark for ONNX, refer to :ref:`link <quark-onnx-quantizing-using-mix-precision>`.

Layer Wise Mixed Precision
--------------------------

Here is a simple example of how to enable layer wise mixed precision in Quark for ONNX.

.. code-block:: python

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

Auto-Mixed Precision
--------------------

Quark for ONNX supports Auto Mixed Precision, which follows the steps below.

1. Quantize the model in wide quantization bits, like 16-bits activation and 8-bits weight. Then run evaluation and get the baseline accuracy.
2. Layers in the model are sorted in the ascending order of the loss sensitivity.
3. Define the quantization target. There are two ways for you to set the accuracy target of Auto Mixed Precision:

   -  Provide the ``Top1AccTarget`` and ``EvaluateFunction``.

      -  ``Top1AccTarget``: The Top1 accuracy loss is no larger than the Top1AccTarget.
      -  ``EvaluateFunction``: The user defined function to calculating the Top1 accuracy of this model.

   -  Provide the target of L2 distance ``L2Target``.

      -  ``L2Target``: The L2 output of the quantized model is no larger than this target.

4. Switch to the narrow quantization bits (like 8-bits) on each layer in the ascending order of the loss of sensitivity, until the user-defined accuracy target is about to be broken.

Here is a simple example of how to enable auto mixed precision in Quark for ONNX:

.. code-block:: python

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
               'WeightTargetQuantType':QuantType.QInt8,
               'OutputIndex': 0,
               'Top1AccTarget': 0.1,
               'EvaluateFunction': top1_acc,
           },
       },
   )
   config = Config(global_quant_config=quant_config)

Quantizing Using Mixed Precision
--------------------------------

Mixed precision improves the quantized model's accuracy by quantizing some nodes with higher precision, though it leads to a loss in performance. The mix-precision options include A16W16_A8W16, A16W16_A16W8, A16W16_A8W8, A16W8_A8W8, and A8W16_A8W8. For example, if A8W8 quantized model's accuracy could not reach your target, you can use the quantization configuration to mix A16W8 and A8W8 as follows:

.. code-block:: python

    from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
    from quark.onnx.quantization.config.config import Config, QuantizationConfig
    import torch

    def get_acc_top1(preds, labels):
        assert len(preds) == len(labels)
        assert len(preds) > 0
        count = 0
        for i in range(len(preds)):
            pred = preds[i]
            label = labels[i]
            if pred == label:
                count += 1
        return count / len(preds)

    def top1_acc(outputs):
        _, preds = torch.max(outputs, 1)
        labels = ['label1', 'label2', 'label3', ...] # label is a list.
        top1_acc_result = get_acc_top1(preds, labels)
        return top1_acc_result

    quant_config = QuantizationConfig(
        calibrate_method=quark.onnx.CalibrationMethod.Percentile,
        quant_format=quark.onnx.VitisQuantFormat.QDQ,
        activation_type=quark.onnx.VitisQuantType.QInt16,
        weight_type=QuantType.QInt8,
        include_auto_mp=True,
        extra_options={
            'ActivationSymmetric': False,
            'WeightsSymmetric': True,
            'Int32Bias': False,
            'AutoMixprecision': {
                'ActTargetQuantType': QuantType.QInt8,
                'WeightTargetQuantType': QuantType.QInt8,
                'OutputIndex': 0,
                'Top1AccTarget': 0.1,
                'EvaluateFunction': top1_acc,
            },
        },
    )
    config = Config(global_quant_config=quant_config)

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

Arguments
~~~~~~~~~

- **quant_format**: (Class) This parameter should be set to ``quark.onnx.VitisQuantFormat.QDQ`` if you use the mixed-precision feature. No default value; you need to specify.

- **activation_type**: (Class) The quant type corresponding to activation in mixed precision has higher or equal precision. No default value; you need to specify.

- **weight_type**: (Class) The quant type corresponding to weight in mixed precision has higher or equal precision. No default value; you need to specify.

- **include_auto_mp**: (Boolean) This parameter is a flag that determines whether to optimize the models using mixed precision; set to True to do mixed precision (default is False).

- **extra_options**: (Dictionary or None) Contains key-value pairs for various options in different cases. Mixed precision-related options are packaged within extra_options as a member whose key is "AutoMixprecision" and values are:

  - **ActTargetQuantType**: (Class) The quant type corresponding to activation in mixed precision has lower or equal precision. No default value; you need to specify.

  - **WeightTargetQuantType**: (Class) The quant type corresponding to weight in mixed precision has lower or equal precision. No default value; you need to specify.

  - **OutputIndex**: (Integer) The index of output to calculate loss between the float model and quantized model. The default value is 0.

  - **Top1AccTarget**: (Float) Top1 accuracy loss that you could accept between the float model and quantized model. No default value; you need to specify.

  - **EvaluateFunction**: (Function) The function to calculate accuracy for the model. The input of the function is model outputs (Tensor), and the output of the function is top1 accuracy (Float). No default function; you need to provide.

Example
--------

For an example of quantizing a `densenet121.ra_in1k` model using mixed precision, refer to this :doc:`Mixed Precision Example <example_quark_onnx_mixed_precision>`.
