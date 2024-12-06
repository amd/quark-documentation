
Microscaling (MX)
=================

In this tutorial, you learn how to use Microscaling (MX) quantization.

MX is an advancement over Block Floating Point (BFP), aiming to improve the
numerical efficiency and flexibility of low-precision computations for AI.

BFP groups numbers (for example, tensors, arrays) into blocks, where each block
shares a common exponent, and the values in the block are represented with
individual mantissas (and the sign bit). This approach is effective for
reducing memory usage, but it is coarse-grained, meaning all numbers within
a block are forced to have the same exponent, regardless of their individual
value ranges.

MX, on the other hand, allows for finer-grained scaling within a block.
Instead of forcing all elements in the block to share a single exponent, MX
assigns a small-scale adjustment to individual or smaller groups of values
within the block. This finer granularity improves precision, as each value
or subgroup of values can adjust more dynamically to their specific range,
reducing the overall quantization error compared to BFP.

What is MX Quantization?
------------------------

The `OCP MX specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
introduces several specific MX formats, including MXFP8, MXFP6, MXFP4, and MXINT8.
These formats are implemented in the Quark ONNX quantizer through a custom operation
named MXFixNeuron, which has an attribute "element_dtype" to set the data type for
the elements (while the data type for the shared scale is always E8M0).

+-------------------+------------------------+
| MX Formats        | "element_dtype" values |
+===================+========================+
| MXFP8(E5M2)       | 'fp8_e5m2'             |
+-------------------+------------------------+
| MXFP8(E4M3)       | 'fp8_e4m3'             |
+-------------------+------------------------+
| MXFP6(E3M2)       | 'fp6_e3m2'             |
+-------------------+------------------------+
| MXFP6(E2M3)       | 'fp6_e2m3'             |
+-------------------+------------------------+
| MXFP4(E2M1)       | 'fp4_e2m1'             |
+-------------------+------------------------+
| MXINT8            | 'int8'                 |
+-------------------+------------------------+

If you initialize the quantizer with the MX configuration, it quantizes all the
activations and weights using the MXFixNeuron.

How to Enable MX Quantization in Quark for ONNX?
------------------------------------------------

Here is a simple example of how to enable MX quantization with MXINT8 in Quark
for ONNX.

.. code:: python

   from quark.onnx import ModelQuantizer, VitisQuantType, VitisQuantFormat
   from onnxruntime.quantization.calibrate import CalibrationMethod
   from quark.onnx.quantization.config.config import Config, QuantizationConfig

   quant_config = QuantizationConfig(calibrate_method=CalibrationMethod.MinMax,
                                     quant_format=VitisQuantFormat.MXFixNeuron,
                                     activation_type=VitisQuantType.QMX,
                                     weight_type=VitisQuantType.QMX,
                                     extra_options={
                                       'MXAttributes': {
                                         'element_dtype': 'int8',
                                         'axis': 1,
                                         'block_size': 8,
                                         'rounding_mode': 2,
                                       },
                                     })

   config = Config(global_quant_config=quant_config)

   quantizer = ModelQuantizer(config)

   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

*Note*: When inferring with ONNX Runtime, you need to register the custom operator's shared object (Linux) or DLL (Windows) file in the ORT session options.

.. code:: python

   import onnxruntime
   from quark.onnx import get_library_path

   if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
       device = 'ROCM'
       providers = ['ROCMExecutionProvider']
   elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
       device = 'CUDA'
       providers = ['CUDAExecutionProvider']
   else:
       device = 'CPU'
       providers = ['CPUExecutionProvider']

   sess_options = onnxruntime.SessionOptions()
   sess_options.register_custom_ops_library(get_library_path(device))
   session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

Example
--------

For an example of quantizing a ResNet50 model using the MX
quantization, refer to the :doc:`MX Example <example_quark_onnx_MX>`.
