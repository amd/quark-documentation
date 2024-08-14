Quark for ONNX - Accuracy Improvement
=====================================

1. Improving accuracy for quantized models
------------------------------------------

quark.onnx provides several techniques to improve the accuracy for
quantized model after PTQ.

1.1 Quantizing Using CrossLayerEqualization(CLE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CrossLayerEqualization (CLE) can equalize the weights of consecutive
convolution layers, making the model weights easier to perform
per-tensor quantization. Experiments show that using CLE technique can
improve the PTQ accuracy of some models, especially for models with
depthwise_conv layers, such as Mobilenet. Here is an example showing how
to enable CLE using quark.onnx.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       quant_format=QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8,
       enable_npu_cnn=True,
       include_cle=True,
       extra_options={
           'ActivationSymmetric':True,
           'ReplaceClip6Relu':True,
           'CLESteps':1,
           'CLEScaleAppendBias':True,
           },
   )
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

**Arguments**

-  **include_cle**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using CrossLayerEqualization; it can
   improve the accuracy of some models. The default is False.

-  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Options related to CLE are:

   -  ReplaceClip6Relu: (Boolean) If True, Replace Clip(0,6) with Relu
      in the model. The default value is False.
   -  CLESteps: (Int) Specifies the steps for CrossLayerEqualization
      execution when include_cle is set to true, The default is 1, When
      set to -1, an adaptive CrossLayerEqualization steps will be
      conducted. The default value is 1.
   -  CLEScaleAppendBias: (Boolean) Whether the bias be included when
      calculating the scale of the weights, The default value is True.

1.2 Quantizing Using Mix Precision
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mix precision improved the quantized model's accuracy by quantizing some
nodes with higher precision, though it leads to a loss in performance.
The mix-precision options: A16W16_A8W16, A16W16_A16W8, A16W16_A8W8,
A16W8_A8W8, A8W16_A8W8. For example, if A8W8 quantized model's accuracy
could not reach your target, you can use the quantization configuration
to mix A16W8 and A8W8 as follows:

.. code:: python

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
           'ActivationSymmetric':False,
           'WeightsSymmetric':True,
           'Int32Bias': False,
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

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

**Arguments**

-  **quant_format**: (Class) This parameter should be set to
   quark.onnx.VitisQuantFormat.QDQ if you use the mix-precision feature.
   No default value; user needs to specify.
-  **activation_type**: (Class) The quant type corresponding to
   activation in mixed precision has higher or equal precision. No
   default value; user needs to specify.
-  **weight_type**: (Class) The quant type corresponding to weight in
   mixed precision has higher or equal precision. No default value; user
   needs to specify.
-  **include_auto_mp**: (Boolean) This parameter is a flag that
   determines whether to optimize the models using mix precision; Set to
   True to do mix precision (default is False).
-  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Mix precision related options are
   packaged within extra_options as a member whose key is
   "AutoMixprecision" and values are:

   -  ActTargetQuantType: (Class) The quant type corresponding to
      activation in mixed precision has lower or equal precision. No
      default value; user needs to specify.
   -  WeightTargetQuantType: (Class) The quant type corresponding to
      weight in mixed precision has lower or equal precision. No default
      value; user needs to specify.
   -  OutputIndex: (Integer) The index of output to caculate loss
      betweenf float model and quantized model. The default value is 0.
   -  Top1AccTarget: (Float) Top1 accuracy loss that user could accept
      between float model and quantized model. No default value; user
      needs to specify.
   -  EvaluateFunction: (Function) The function to caculate accuracy for
      the model. Input of the function is model outputs(Tensor), output
      of the function is top1 accuracy(Float). No default function; user
      needs to provide.

1.3 Quantizing Using Fast Finetune
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast finetune improves the quantized model's accuracy by training the
output of each layer as close as possible to the floating-point model.
It includes two practical algorithms "AdaRound" and "AdaQuant". Applying
fast finetune may get better accuracy for some models but will take much
longer time than normal PTQ. It is disabled by default to save
quantization time but can be turned on if you see accuracy issues. Note
that once enabled this feature, the quark.onnx will require PyTorch
package.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       quant_format=QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8,
       enable_npu_cnn=True,
       include_fast_ft=True,
       extra_options={
           'ActivationSymmetric':True,
           'FastFinetune': {
               'OptimAlgorithm':'adaround',
               'OptimDevice':'cpu',
               'BatchSize':1,
               'NumIterations':1000,
               'LearningRate':0.1,
           },
       },
   )
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

**Arguments**

-  **include_fast_ft**: (Boolean) This parameter is a flag that
   determines whether to optimize the models using Fast Finetune; Set to
   True to do fast finetune (default is False).
-  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Fast finetune related options are
   packaged within extra_options as a member whose key is "FastFinetune"
   and values are:

   -  OptimAlgorithm: (String) The specified algorithm for fast
      finetune. Optional values are "adaround" and "adaquant", the
      former adjusts the weight's rounding function, which is relatively
      stable and might converge faster, while the latter trains the
      weight directly, so might have a greater improvement. The default
      value is "adaround".
   -  OptimDevice: (String) The compute device for fast finetune.
      Optional values are "cpu", "hip:0" and "cuda:0". The default value
      is "cpu".
   -  BatchSize: (Int) Batch size for finetuning. The larger batch size,
      the better accuracy but the longer training time. The default
      value is 1.
   -  NumIterations: (Int) The Iterations for finetuning. The more
      iterations, the better accuracy but the longer training time. The
      default value is 1000.
   -  LearningRate: (Float) Learning rate for finetuning. It has a
      significant impact on the improvement of fast finetune, you need
      to try some learning rates to get a better result for your model.
      The default value is 0.1.

1.4 Quantizing Using SmoothQuant(SQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SmoothQuant(SQ) is another technique used to improve PTQ accuracy. It
smoothes the outliers of the activation so that it loses as little
precision as possible during quantization. Experiments show that using
SQ technique can improve the PTQ accuracy of some models, especially for
models with a large number of outliers in the activation. Here is an
example showing how to enable SQ using quark.onnx.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       quant_format=QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       activation_type=QuantType.QUInt8,
       weight_type=QuantType.QInt8,
       enable_npu_cnn=True,
       include_sq=True,
       extra_options={
           'ActivationSymmetric':True,
           'SmoothAlpha':0.5,
           },
   )
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

**Arguments**

-  **include_sq**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using SmoothQuant; it can improve the
   accuracy of some models. The default is False.

-  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Options related to SQ are:

   -  SmoothAlpha: (Float) This parameter control how much difficulty we
      want to migrate from activation to weights, The default value is
      0.5.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
