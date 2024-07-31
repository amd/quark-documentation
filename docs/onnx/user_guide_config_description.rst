Quark for ONNX - Configuration Description
==========================================

Configurations
--------------

Configuration of quantization in ``Quark for ONNX`` is set by python
``dataclass`` because it is rigorous and can help users avoid typos. We
provide a class ``Config`` in ``quark.onnx.quantization.config.config``
for configuration, as demonstrated in the example above. In ``Config``,
users should set certain instances (all instances are optional except
global_quant_CONFIG:):

-  ``global_quant_CONFIG:(QuantizationConfig)``: Global quantization
   configuration applied to the entire model.

The ``Config`` should be like:

.. code:: python

   from quark.onnx.quantization.config.config import Config
   quant_config = Config(global_quant_config=...)

We defined some default global configrations, including
``DEFAULT_XINT8_CONFIG`` and ``DEFAULT_U8S8_AAWS_CONFIG``, which can be
used like this:

.. code:: python

   quant_config = Config(global_quant_config=DEFAULT_U8S8_AAWS_CONFIG)

More Quantization Default Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark for ONNX provides user with the default configurations shown
below.

-  ``DEFAULT_XINT8_CONFIG``: Perform uint8 activation, int8 weight,
   optimized for NPU quantization.
-  ``DEFAULT_XINT8_ADAROUND_CONFIG``: Perform uint8 activation, int8
   weight, optimized for NPU quantization. The adaround fast finetune
   will be applied to perserve quantized accuracy.
-  ``DEFAULT_XINT8_ADAQUANT_CONFIG``: Perform uint8 activation, int8
   weight, optimized for NPU quantization. The adaquant fast finetune
   will be applied to perserve quantized accuracy.
-  ``DEFAULT_S8S8_AAWS_CONFIG``: Perform int8 asymmetric activation,
   int8 symmetric weight quantization.
-  ``DEFAULT_S8S8_AAWS_ADAROUND_CONFIG``: Perform int8 asymmetric
   activation, int8 symmetric weight quantization. The adaround fast
   finetune will be applied to perserve quantized accuracy.
-  ``DEFAULT_S8S8_AAWS_ADAQUANT_CONFIG``: Perform int8 asymmetric
   activation, int8 symmetric weight quantization. The adaquant fast
   finetune will be applied to perserve quantized accuracy.
-  ``DEFAULT_U8S8_AAWS_CONFIG``: Perform uint8 asymmetric activation,
   int8 symmetric weight quantization.
-  ``DEFAULT_U8S8_AAWS_ADAROUND_CONFIG``:
   Perform uint8 asymmetric activation, int8 symmetric weight
   quantization. The adaround fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_U8S8_AAWS_ADAQUANT_CONFIG``:
   Perform uint8 asymmetric activation, int8 symmetric weight
   quantization. The adaquant fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_S16S8_ASWS_CONFIG``:
   Perform int16 symmetric activation, int8 symmetric weight
   quantization.
-  ``DEFAULT_S16S8_ASWS_ADAROUND_CONFIG``:
   Perform int16 symmetric activation, int8 symmetric weight
   quantization. The adaround fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_S16S8_ASWS_ADAQUANT_CONFIG``:
   Perform int16 symmetric activation, int8 symmetric weight
   quantization. The adaquant fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_U16S8_AAWS_CONFIG``:
   Perform uint16 asymmetric activation, int8 symmetric weight
   quantization.
-  ``DEFAULT_U16S8_AAWS_ADAROUND_CONFIG``:
   Perform uint16 asymmetric activation, int8 symmetric weight
   quantization. The adaround fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_U16S8_AAWS_ADAQUANT_CONFIG``:
   Perform uint16 asymmetric activation, int8 symmetric weight
   quantization. The adaquant fast finetune will be applied to perserve
   quantized accuracy.
-  ``DEFAULT_BF16_CONFIG``:
   Perform bfloat16 activation, bfloat16 weight quantization.
-  ``DEFAULT_BFP16_CONFIG``:
   Perform BFP16 activation, BFP16 weight quantization.
-  ``DEFAULT_S16S16_MIXED_S8S8_CONFIG``:
   Perform int16 activation, int16 weight mix-percision quantization.

Customized Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

Besides the default configurations in Quark ONNX, user can also
customize the quantization configuration like the example below. Please
refer to `Full List of Quantization Config
Features <./appendix_full_quant_config_features.html>`__ for more details.

.. code:: python

   from quark.onnx import ModelQuantizer, PowerOfTwoMethod, QuantType
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(
       quant_format=quark.onnx.QuantFormat.QDQ,
       calibrate_method=quark.onnx.PowerOfTwoMethod.MinMSE,
       input_nodes=[],
       output_nodes=[],
       op_types_to_quantize=[],
       random_data_reader_input_shape=[],
       per_channel=False,
       reduce_range=False,
       activation_type=quark.onnx.QuantType.QInt8,
       weight_type=quark.onnx.QuantType.QInt8,
       nodes_to_quantize=[],
       nodes_to_exclude=[],
       optimize_model=True,
       use_external_data_format=False,
       execution_providers=['CPUExecutionProvider'],
       enable_npu_cnn=False,
       enable_npu_transformer=False,
       convert_fp16_to_fp32=False,
       convert_nchw_to_nhwc=False,
       include_cle=False,
       include_sq=False,
       extra_options={},)
   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)
   quantizer.quantize_model(input_model_path, output_model_path, calibration_data_reader=None)

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
