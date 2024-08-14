Quark for ONNX - Optional Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Exporting PyTorch Models to ONNX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Please skip this step if you already have the onnx format model.**

For PyTorch models, it is recommended to use the TorchScript-based onnx
exporter for exporting ONNX models. Please refer to the `PyTorch
documentation for
guidance <https://pytorch.org/docs/stable/onnx_torchscript.html#torchscript-based-onnx-exporter>`__.

Tips: 1. Before exporting, please perform the model.eval(). 2. Models
with opset 17 are recommended. 3. For NPU_CNN platforms, dynamic input
shapes are currently not supported and only a batch size of 1 is
allowed. Please ensure that the shape of input is a fixed value, and the
batch dimension is set to 1.

Example code:

.. code:: python

   torch.onnx.export(
       model,
       input,
       model_output_path,
       opset_version=17,
       input_names=['input'],
       output_names=['output'],
   )

-  **Opset Versions**: Models with opset 17 are recommended. Models must
   be opset 10 or higher to be quantized. Models with opset lower than
   10 should be reconverted to ONNX from their original framework using
   a later opset. Alternatively, you can refer to the usage of the
   version converter for `ONNX Version
   Converter <https://github.com/onnx/onnx/blob/main/docs/VersionConverter.html>`__.
   Opset 10 does not support some node fusions and may not get the best
   performance. We recommend to update the model to opset 17 for better
   performance. Moreover, per channel quantization is supported for
   opset 13 or higher versions.

-  **Large Models > 2GB**: Due to the 2GB file size limit of Protobuf,
   for ONNX models exceeding 2GB, additional data will be stored
   separately. Please ensure that the .onnx file and the data file are
   placed in the same directory. Also, please set the
   use_external_data_format parameter to True for large models when
   quantizing.

2. Pre-processing on the Float Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pre-processing is to transform a float model to prepare it for
quantization. It consists of the following three optional steps:

-  Symbolic shape inference: This is best suited for transformer models.
-  Model Optimization: This step uses ONNX Runtime native library to
   rewrite the computation graph, including merging computation nodes,
   and eliminating redundancies to improve runtime efficiency.
-  ONNX shape inference.

The goal of these steps is to improve quantization quality. ONNX Runtime
quantization tool works best when the tensor's shape is known. Both
symbolic shape inference and ONNX shape inference help figure out tensor
shapes. Symbolic shape inference works best with transformer-based
models, and ONNX shape inference works with other models.

Model optimization performs certain operator fusion that makes the
quantization tool's job easier. For instance, a Convolution operator
followed by BatchNormalization can be fused into one during the
optimization, which can be quantized very efficiently.

Unfortunately, a known issue in ONNX Runtime is that model optimization
can not output a model size greater than 2GB. So for large models,
optimization must be skipped.

Pre-processing API is in the Python module
onnxruntime.quantization.shape_inference, function quant_pre_process().

.. code:: python

   from onnxruntime.quantization import shape_inference

   shape_inference.quant_pre_process(
        input_model_path: str,
       output_model_path: str,
       skip_optimization: bool = False,
       skip_onnx_shape: bool = False,
       skip_symbolic_shape: bool = False,
       auto_merge: bool = False,
       int_max: int = 2**31 - 1,
       guess_output_rank: bool = False,
       verbose: int = 0,
       save_as_external_data: bool = False,
       all_tensors_to_one_file: bool = False,
       external_data_location: str = "./",
       external_data_size_threshold: int = 1024,)

**Arguments**

-  **input_model_path**: (String) This parameter specifies the file path
   of the input model that is to be pre-processed for quantization.
-  **output_model_path**: (String) This parameter specifies the file
   path where the pre-processed model will be saved.
-  **skip_optimization**: (Boolean) This flag indicates whether to skip
   the model optimization step. If set to True, model optimization will
   be skipped, which may cause ONNX shape inference failure for some
   models. The default value is False.
-  **skip_onnx_shape**: (Boolean) This flag indicates whether to skip
   the ONNX shape inference step. The symbolic shape inference is most
   effective with transformer-based models. Skipping all shape
   inferences may reduce the effectiveness of quantization, as a tensor
   with an unknown shape cannot be quantized. The default value is
   False.
-  **skip_symbolic_shape**: (Boolean) This flag indicates whether to
   skip the symbolic shape inference step. Symbolic shape inference is
   most effective with transformer-based models. Skipping all shape
   inferences may reduce the effectiveness of quantization, as a tensor
   with an unknown shape cannot be quantized. The default value is
   False.
-  **auto_merge**: (Boolean) This flag determines whether to
   automatically merge symbolic dimensions when a conflict occurs during
   symbolic shape inference. The default value is False.
-  **int_max**: (Integer) This parameter specifies the maximum integer
   value that is to be considered as boundless for operations like slice
   during symbolic shape inference. The default value is 2**31 - 1.
-  **guess_output_rank**: (Boolean) This flag indicates whether to guess
   the output rank to be the same as input 0 for unknown operations. The
   default value is False.
-  **verbose**: (Integer) This parameter controls the level of detailed
   information logged during inference. A value of 0 turns off logging,
   1 logs warnings, and 3 logs detailed information. The default value
   is 0.
-  **save_as_external_data**: (Boolean) This flag determines whether to
   save the ONNX model to external data. The default value is False.
-  **all_tensors_to_one_file**: (Boolean) This flag indicates whether to
   save all the external data to one file. The default value is False.
-  **external_data_location**: (String) This parameter specifies the
   file location where the external file is saved. The default value is
   "./".
-  **external_data_size_threshold**: (Integer) This parameter specifies
   the size threshold for external data. The default value is 1024.

2. Evaluating the Quantized Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have scripts to evaluate float models, like the models in Xilinx
Model Zoo, you can replace the float model file with the quantized model
for evaluation. Note that if customized Q/DQ is used in the quantized
model, it is necessary to register the custom operations library to
onnxruntime inference session before evaluation. For example:

.. code:: python

   import onnxruntime as ort

   so = ort.SessionOptions()
   so.register_custom_ops_library(quark.onnx.get_library_path())
   sess = ort.InferenceSession(quantized_model, so)

3. Dumping the Simulation Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes after deploying the quantized model, it is necessary to
compare the simulation results on the CPU/GPU and the output values on
the DPU. You can use the dump_model API of Quark ONNX to dump the
simulation results with the quantized_model. Currently, only models
containing FixNeuron nodes support this feature. For models using
QuantFormat.QDQ, you can set 'dump_float' to True to save float data for
all nodes' results.

.. code:: python

   # This function dumps the simulation results of the quantized model,
   # including weights and activation results.
   quark.onnx.dump_model(
       model,
       dump_data_reader=None,
       random_data_reader_input_shape=[],
       dump_float=False,
       output_dir='./dump_results',)

**Arguments**

-  **model**: (String) This parameter specifies the file path of the
   quantized model whose simulation results are to be dumped.
-  **dump_data_reader**: (CalibrationDataReader or None) This parameter
   is a data reader that is used for the dumping process. The first
   batch will be taken as input. If you wish to use random data for a
   quick test, you can set dump_data_reader to None. The default value
   is None.
-  **random_data_reader_input_shape**: (List or Tuple of Int) If dynamic
   axes of inputs require specific value, users should provide its
   shapes when using internal random data reader (That is, set
   dump_data_reader to None). The basic format of shape for single input
   is list (Int) or tuple (Int) and all dimensions should have concrete
   values (batch dimensions can be set to 1). For example,
   random_data_reader_input_shape=[1, 3, 224, 224] or
   random_data_reader_input_shape=(1, 3, 224, 224) for single input. If
   the model has multiple inputs, it can be fed in list (shape) format,
   where the list order is the same as the onnxruntime got inputs. For
   example, random_data_reader_input_shape=[[1, 1, 224, 224], [1, 2,
   224, 224]] for 2 inputs. Moreover, it is possible to use dict {name :
   shape} to specify a certain input, for example,
   random_data_reader_input_shape={"image" : [1, 3, 224, 224]} for the
   input named "image". The default value is [].
-  **dump_float**: (Boolean) This flag determines whether to dump the
   floating-point value of nodes' results. If set to True, the float
   values will be dumped. Note that this may require a lot of storage
   space. The default value is False.
-  **output_dir**: (String) This parameter specifies the directory where
   the dumped simulation results will be saved. After successful
   execution of the function, dump results are generated in this
   specified directory. The default value is './dump_results'.

Note: The batch_size of the dump_data_reader will be better to set to 1
for DPU debugging.

Dump results of each FixNeuron node (including weights and activation)
are generated in output_dir after the command has been successfully
executed.

For each quantized node, results are saved in *.bin and* .txt formats
(\* represents the output name of the node). If "dump_float" is set to
True, output of all nodes are saved in \*_float.bin and \*_float.txt (\*
represents the output name of the node), please note that this may
require a lot of storage space.

Examples of dumping results are shown in the following table. Due to
considerations for the storage path, the '/' in the node name will be
replaced with '\_'.

Table 2. Example of Dumping Results

+------------------------+------------------------+--------------------+
| Quantized              | Node Name              | Saved Weights or   |
|                        |                        | Activations        |
+========================+========================+====================+
| Yes                    | /conv1/Conv_out        | {ou                |
|                        | put_0_DequantizeLinear | tput_dir}/dump_res |
|                        |                        | ults/\_conv1_Conv_ |
|                        |                        | output_0_Dequantiz |
|                        |                        | eLinear_Output.bin |
|                        |                        | {ou                |
|                        |                        | tput_dir}/dump_res |
|                        |                        | ults/\_conv1_Conv_ |
|                        |                        | output_0_Dequantiz |
|                        |                        | eLinear_Output.txt |
+------------------------+------------------------+--------------------+
| Yes                    | onnx::Con              | {output_dir}/d     |
|                        | v_501_DequantizeLinear | ump_results/onnx:: |
|                        |                        | Conv_501_Dequantiz |
|                        |                        | eLinear_Output.bin |
|                        |                        | {output_dir}/d     |
|                        |                        | ump_results/onnx:: |
|                        |                        | Conv_501_Dequantiz |
|                        |                        | eLinear_Output.txt |
+------------------------+------------------------+--------------------+
| No                     | /avg                   | {output_dir}/dump_ |
|                        | pool/GlobalAveragePool | results/\_avgpool_ |
|                        |                        | GlobalAveragePool_ |
|                        |                        | output_0_float.bin |
|                        |                        | {output_dir}/dump_ |
|                        |                        | results/\_avgpool_ |
|                        |                        | GlobalAveragePool_ |
|                        |                        | output_0_float.txt |
+------------------------+------------------------+--------------------+

.. raw:: html

   <!--
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
