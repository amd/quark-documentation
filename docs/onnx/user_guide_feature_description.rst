Quark for ONNX - Feature Description
====================================

Quantization Configuration Key Features
---------------------------------------

Quark for ONNX provides the key features as below:

+--------------------+-------------------------------------------------+
| Feature Name       | Feature Value                                   |
+====================+=================================================+
| Activation/Weight  | Int8 / Uint8/ Int16 / Uint16 / Int32 / Uint32 / |
| Type               | `Float16 <https://en.wikipedia.or               |
|                    | g/wiki/Half-precision_floating-point_format>`__ |
|                    | /                                               |
|                    | `Bfloat16 <https://en.wikipe                    |
|                    | dia.org/wiki/Bfloat16_floating-point_format>`__ |
+--------------------+-------------------------------------------------+
| Quant Strategy     | Static quant / Weight only / Dynamic quant      |
+--------------------+-------------------------------------------------+
| Quant Scheme       | Per tensor / Per channel                        |
+--------------------+-------------------------------------------------+
| Quant Format       | QuantFormatQDQ / VitisQuantFormat.QDQ /         |
|                    | QuantFormat.QOperator                           |
+--------------------+-------------------------------------------------+
| Calibration method | MinMax / Percentile / MinMSE / Entropy /        |
|                    | NonOverflow                                     |
+--------------------+-------------------------------------------------+
| Symmetric          | Symmetric / Asymmetric                          |
+--------------------+-------------------------------------------------+
| Scale Type         | Float32 / Float16                               |
+--------------------+-------------------------------------------------+
| Pre-Quant          | SmoothQuant (Single_GPU/CPU) / CLE / Bias       |
| Optimization       | Correction                                      |
+--------------------+-------------------------------------------------+
| Quant Algorithm    | AdaQuant / AdaRound / GPTQ                            |
+--------------------+-------------------------------------------------+
| Operating Systems  | Linux(ROCm/CUDA) / Windows(CPU)                 |
+--------------------+-------------------------------------------------+

We present detailed explanations of these features:

Quant Strategy
~~~~~~~~~~~~~~

Quark for ONNX offers three distinct quantization strategies tailored to
meet the requirements of various HW backends:

-  **Post Training Weight-Only Quantization**: The weights are quantized
   ahead of time but the activations are not quantized(using original
   float data type) during inference.

-  **Post Training Static Quantization**: Post Training Static
   Quantization quantizes both the weights and activations in the model.
   To achieve the best results, this process necessitates calibration
   with a dataset that accurately represents the actual data, which
   allows for precise determination of the optimal quantization
   parameters for activations.

- **Post Training Dynamic Quantization**: Dynamic Quantization quantizes
   the weights ahead of time, while the activations are quantized
   dynamically at runtime. This method allows for a more flexible
   approach, especially when the activation distribution is not
   well-known or varies significantly during inference.

The strategies share the same user API. Users simply need to set the
strategy through the quantization configuration, as demonstrated in the
example above. More details about setting quantization configuration are
in the chapter "Configuring Quark for ONNX"

The Quant Schemes
~~~~~~~~~~~~~~~~~

Quark for ONNX is capable of handling ``per tensor`` and ``per channel``
quantization, supporting both symmetric and asymmetric methods.

-  **Per Tensor Quantization** means that quantize the tensor with one
   scalar. The scaling factor is a scalar.

-  **Per Channel Quantization** means that for each dimension, typically
   the channel dimension of a tensor, the values in the tensor are
   quantized with different quantization parameters. The scaling factor
   is a 1-D tensor, with the length of the quantization axis. For the
   input tensor with shape ``(D0, ..., Di, ..., Dn)`` and ``ch_axis=i``,
   The scaling factor is a 1-D tensor of length ``Di``.

The Symmetric/Asymmetric Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Symmetric/Asymmetric quantization`` is primarily used to describe the
quantization of integers. ``Symmetric quantization`` involves scaling
the data by a fixed scaling factor, and zero-point is generally set at
zero. ``Asymmetric quantization`` uses a scaling factor and a zero-point
that can shift, allowing the zero of the quantized data to represent a
value other than zero.

The Calibration Methods
~~~~~~~~~~~~~~~~~~~~~~~

Quark for PyTorch supports these types of calibration methods:

-  **MinMax Calibration method**: The ``MinMax`` calibration method for
   computing the quantization parameters based on the running min and
   max values. This method uses the tensor min/max statistics to compute
   the quantization parameters. The module records the running minimum
   and maximum of incoming tensors and uses this statistic to compute
   the quantization parameters.

-  **Percentile Calibration method**: The ``Percentile`` calibration
   method, often used in robust scaling, involves scaling features based
   on percentile information from a static histogram, rather than using
   the absolute minimum and maximum values. This method is particularly
   useful for managing outliers in data.

-  **MSE Calibration method**: The ``MSE`` (Mean Squared Error)
   calibration method refers to a method where calibration is performed
   by minimizing the mean squared error between the predicted outputs
   and the actual outputs. This method is typically used in regression
   contexts where the goal is to adjust model parameters or data
   transformations to reduce the average squared difference between
   estimated values and the true values. MSE calibration helps in
   refining model accuracy by fine-tuning predictions to be as close as
   possible to the real data points.

-  **Entropy Calibration Method**: The Entropy calibration method refers
   to a method determines he quantization parameters by considering the
   entropy algorithm of each tensor's distribution.

-  **NonOverflow Calibration Method**: The NonOverflow calibration
   method gets the power-of-two quantization parameters for each tensor
   to make sure min/max values not overflow.

Pre-Quant Optimization
~~~~~~~~~~~~~~~~~~~~~~

Quark for ONNX supports ``SmoothQuant``, ``CLE``\ (Cross Layer
Equalization) and ``Bias Correction`` as the pre-quant optimization.

Quant Algorithm
~~~~~~~~~~~~~~~

Quark for ONNX supports ``AdaQuant``, ``AdaRound`` and ``GPTQ`` as the quant
algorithms.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
