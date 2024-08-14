Full List of Quantization Configuration Features
================================================

Quantization Configuration

.. code:: python

   from quark.onnx import QuantType
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

**Arguments**

*  **model_input**: (String) This parameter specifies the file path of the model that is to be quantized.
*  **model_output**: (String) This parameter specifies the file path where the quantized model will be saved. 
*  **calibration_data_reader**: (Object or None) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. If you wish to use random data for a quick test, you can set calibration_data_reader to None. 
*  **quant_format**: (String) This parameter is used to specify the quantization format of the model. It has the following options: 

   -  quark.onnx.QuantFormat.QOperator: This option quantizes the model directly using quantized operators. 
   -  quark.onnx.QuantFormat.QDQ: This option quantizes the model by inserting QuantizeLinear/DeQuantizeLinear into the tensor. It supports 8-bit quantization only. 
   -  quark.onnx.VitisQuantFormat.QDQ: This option quantizes the model by inserting VitisQuantizeLinear/VitisDequantizeLinear into the tensor. It supports a wider range of bit-widths and precisions. 
   -  quark.onnx.VitisQuantFormat.FixNeuron (Experimental): This option quantizes the model by inserting FixNeuron (a combination of QuantizeLinear and DeQuantizeLinear) into the tensor. This quant format is currently experimental and cannot use for actual deployment. 

*  **calibrate_method**: (String) The method used in calibration, default to quark.onnx.PowerOfTwoMethod.MinMSE.
   
   For NPU_CNN platforms, power-of-two methods should be used, options are:
   
   -  quark.onnx.PowerOfTwoMethod.NonOverflow: This method get the power-of-two quantize parameters for each tensor to make sure min/max values not overflow.
   -  quark.onnx.PowerOfTwoMethod.MinMSE: This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values. This takes longer time but usually gets better accuracy.
   
   For NPU_Transformer or CPU platforms, float scale methods should be used, options are:
   
   -  quark.onnx.CalibrationMethod.MinMax: This method obtains the
      quantization parameters based on the minimum and maximum values of
      each tensor.
   -  quark.onnx.CalibrationMethod.Entropy: This method determines the
      quantization parameters by considering the entropy algorithm of each
      tensor's distribution.
   -  quark.onnx.CalibrationMethod.Percentile: This method calculates 
      quantization parameters using percentiles of the tensor values.
*  **input_nodes**: (List of Strings) This parameter is a list of the
   names of the starting nodes to be quantized. Nodes in the model
   before these nodes will not be quantized. For example, this argument
   can be used to skip some pre-processing nodes or stop the first node
   from being quantized. The default value is an empty list ([]).
*  **output_nodes**: (List of Strings) This parameter is a list of the
   names of the end nodes to be quantized. Nodes in the model after
   these nodes will not be quantized. For example, this argument can be
   used to skip some post-processing nodes or stop the last node from
   being quantized. The default value is an empty list ([]).
*  **op_types_to_quantize**: (List of Strings or None) If specified,
   only operators of the given types will be quantized (e.g., ['Conv']
   to only quantize Convolutional layers). By default, all supported
   operators will be quantized.
*  **random_data_reader_input_shape**: (List or Tuple of Int) If dynamic
   axes of inputs require specific value, users should provide its
   shapes when using internal random data reader (That is, set
   calibration_data_reader to None). The basic format of shape for
   single input is list (Int) or tuple (Int) and all dimensions should
   have concrete values (batch dimensions can be set to 1). For example,
   random_data_reader_input_shape=[1, 3, 224, 224] or
   random_data_reader_input_shape=(1, 3, 224, 224) for single input. If
   the model has multiple inputs, it can be fed in list (shape) format,
   where the list order is the same as the onnxruntime got inputs. For
   example, random_data_reader_input_shape=[[1, 1, 224, 224], [1, 2,
   224, 224]] for 2 inputs. Moreover, it is possible to use dict {name :
   shape} to specify a certain input, for example,
   random_data_reader_input_shape={“image” : [1, 3, 224, 224]} for the
   input named “image”. The default value is an empty list ([]).
-  **per_channel**: (Boolean) Determines whether weights should be
   quantized per channel. The default value is False. For DPU/NPU
   devices, this must be set to False as they currently do not support
   per-channel quantization.
-  **reduce_range**: (Boolean) If True, quantizes weights with 7-bits.
   The default value is False. For DPU/NPU devices, this must be set to
   False as they currently do not support reduced range quantization.
-  **activation_type**: (QuantType) Specifies the quantization data type
   for activations, options can be found in the table below. The default
   is quark.onnx.QuantType.QInt8.
-  **weight_type**: (QuantType) Specifies the quantization data type for
   weights, options can be found in the table below. The default is
   quark.onnx.QuantType.QInt8. For NPU devices, this must be set to
   QuantType.QInt8.
-  **nodes_to_quantize**:(List of Strings or None) If specified, only
   the nodes in this list are quantized. The list should contain the
   names of the nodes, for example, ['Conv\__224', 'Conv\__252']. The
   default value is an empty list ([]).
-  **nodes_to_exclude**:(List of Strings or None) If specified, the
   nodes in this list will be excluded from quantization. The default
   value is an empty list ([]).
-  **optimize_model**:(Boolean) If True, optimizes the model before
   quantization. Model optimization performs certain operator fusion
   that makes quantization tool's job easier. For instance, a
   Conv/ConvTranspose/Gemm operator followed by BatchNormalization can
   be fused into one during the optimization, which can be quantized
   very efficiently. The default value is True.
-  **use_external_data_format**: (Boolean) This option is used for large
   size (>2GB) model. The model proto and data will be stored in
   separate files. The default is False.
-  **execution_providers**: (List of Strings) This parameter defines the
   execution providers that will be used by ONNX Runtime to do
   calibration for the specified model. The default value
   'CPUExecutionProvider' implies that the model will be computed using
   the CPU as the execution provider. You can also set this to other
   execution providers supported by ONNX Runtime such as
   'CUDAExecutionProvider' for GPU-based computation, if they are
   available in your environment. The default is
   ['CPUExecutionProvider'].
-  **enable_npu_cnn**: (Boolean) This parameter is a flag that
   determines whether to generate a quantized model that is suitable for
   the DPU/NPU. If set to True, the quantization process will consider
   the specific limitations and requirements of the DPU/NPU, thus
   creating a model that is optimized for DPU/NPU computations. This
   parameter primarily addresses the optimization of CNN based models
   for deployment on DPU/NPU. The default is False. **Note**: In the
   previous versions, "enable_npu_cnn" was named "enable_dpu".
   "enable_dpu" will be deprecated in future releases, please use
   "enable_npu_cnn" instead.
-  **enable_npu_transformer**: (Boolean) This parameter is a flag that
   determines whether to generate a quantized model that is suitable for
   the NPU. If set to True, the quantization process will consider the
   specific limitations and requirements of the NPU, thus creating a
   model that is optimized for NPU computations. This parameter
   primarily addresses the optimization of transformer models for
   deployment on NPU. The default is False.
-  **convert_fp16_to_fp32**: (Boolean) This parameter controls whether
   to convert the input model from float16 to float32 before
   quantization. For float16 models, it is recommended to set this
   parameter to True. The default value is False. When using
   convert_fp16_to_fp32 in Quark for ONNX, it requires onnxsim to
   simplify the ONNX model. Please make sure that onnxsim is installed
   by using 'python -m pip install onnxsim'.
-  **convert_nchw_to_nhwc**: (Boolean) This parameter controls whether
   to convert the input NCHW model to input NHWC model before
   quantization. For input NCHW models, it is recommended to set this
   parameter to True. If you provide a custom calibration_data_reader,
   its shape needs to be nhwc instead of nchw when this parameter is set
   to True. The default value is False.
-  **include_cle**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using CrossLayerEqualization; it can
   improve the accuracy of some models. The default is False.
-  **include_fast_ft**: (Boolean) This parameter is a flag that
   determines whether to use adaround or adaquant algorithm for
   finetuning, this is an experimental feature. The default is False.
-  **include_sq**: (Boolean) This parameter is a flag that determines
   whether to optimize the models using SmoothQuant; it can improve the
   accuracy of some models. The default is False.
-  **specific_tensor_precision**: (Boolean) This parameter is a flag
   that determines whether to use tensor-level mixed precision, this is
   an experimental feature. The default is False.
-  **extra_options**: (Dictionary or None) Contains key-value pairs for
   various options in different cases. Current used:

   -  **ActivationSymmetric**: (Boolean) If True, symmetrize calibration
      data for activations. The default is False.
   -  **WeightSymmetric**: (Boolean) If True, symmetrize calibration
      data for weights. The default is True.
   -  **UseUnsignedReLU**: (Boolean) If True, the output tensor of ReLU
      and Clip, whose min is 0, will be forced to be asymmetric. The
      default is False.
   -  **QuantizeBias**: (Boolean) If True, quantize the Bias as a normal
      weights. The default is True. For DPU/NPU devices, this must be
      set to True.
   -  **Int32Bias**: (Boolean) If True, bias will be quantized in int32
      datatype; if false, it will have the same datatype as weight. The
      default is False when enable_npu_cnn is True. Otherwise the
      default is True.
   -  **RemoveInputInit**: (Boolean) If True, initializer in graph
      inputs will be removed because it will not be treated as constant
      value/weight. This may prevent some of the graph optimizations,
      like const folding. The default is True.
   -  **SimplifyModel**: (Boolean) If True, The input model will be
      simplified using the onnxsim tool. The default is True.
   -  **EnableSubgraph**: (Boolean) If True, the subgraph will be
      quantized. The default is False. More support for this feature is
      planned in the future.
   -  **ForceQuantizeNoInputCheck**: (Boolean) If True, latent operators
      such as maxpool and transpose will always quantize their inputs,
      generating quantized outputs even if their inputs have not been
      quantized. The default behavior can be overridden for specific
      nodes using nodes_to_exclude.
   -  **MatMulConstBOnly**: (Boolean) If True, only MatMul operations
      with a constant 'B' will be quantized. The default is False.
   -  **AddQDQPairToWeight**: (Boolean) If True, both QuantizeLinear and
      DeQuantizeLinear nodes are inserted for weight, maintaining its
      floating-point format. The default is False, which quantizes
      floating-point weight and feeds it solely to an inserted
      DeQuantizeLinear node. In the PowerOfTwoMethod calibration method,
      this setting will also be effective for the bias.
   -  **OpTypesToExcludeOutputQuantization**: (List of Strings or None)
      If specified, the output of operators with these types will not be
      quantized. The default is an empty list.
   -  **DedicatedQDQPair**: (Boolean) If True, an identical and
      dedicated QDQ pair is created for each node. The default is False,
      allowing multiple nodes to share a single QDQ pair as their
      inputs.
   -  **QDQOpTypePerChannelSupportToAxis**: (Dictionary) Sets the
      channel axis for specific operator types (e.g., {'MatMul': 1}).
      This is only effective when per-channel quantization is supported
      and per_channel is True. If a specific operator type supports
      per-channel quantization but no channel axis is explicitly
      specified, the default channel axis will be used. For DPU/NPU
      devices, this must be set to {} as per-channel quantization is
      currently unsupported. The default is an empty dict ({}).
   -  **UseQDQVitisCustomOps**: (Boolean) If True, The UInt8 and Int8
      quantization will be executed by the custom operations library,
      otherwise by the library of onnxruntime extensions. The default is
      True, only valid in quark.onnx.VitisQuantFormat.QDQ.
   -  **CalibTensorRangeSymmetric**: (Boolean) If True, the final range
      of the tensor during calibration will be symmetrically set around
      the central point "0". The default is False. In PowerOfTwoMethod
      calibration method, the default is True.
   -  **CalibMovingAverage**: (Boolean) If True, the moving average of
      the minimum and maximum values will be computed when the
      calibration method selected is MinMax. The default is False. In
      PowerOfTwoMethod calibration method, this should be set to False.
   -  **CalibMovingAverageConstant**: (Float) Specifies the constant
      smoothing factor to use when computing the moving average of the
      minimum and maximum values. The default is 0.01. This is only
      effective when the calibration method selected is MinMax and
      CalibMovingAverage is set to True. In PowerOfTwoMethod calibration
      method, this option is unsupported.
   -  **Percentile**: (Float) If the calibration method is set to
      'quark.onnx.CalibrationMethod.Percentile,' then this parameter can
      be set to the percentage for percentile. The default is 99.999.
   -  **RandomDataReaderInputDataRange**: (Dict or None) Specifies the
      data range for each inputs if used random data reader
      (calibration_data_reader is None). Currently, if set to None then
      the random value will be 0 or 1 for all inputs, otherwise range
      [-128,127] for unsigned int, range [0,255] for signed int and
      range [0,1] for other float inputs. The default is None.
   -  **Int16Scale**: (Boolean) If True, the float scale will be
      replaced by the closest value corresponding to M and 2\ **N, where
      the range of M and 2**\ N is within the representation range of
      int16 and uint16. The default is False.
   -  **MinMSEMode**: (String) When using
      quark.onnx.PowerOfTwoMethod.MinMSE, you can specify the method for
      calculating minmse. By default, minmse is calculated using all
      calibration data. Alternatively, you can set the mode to
      "MostCommon", where minmse is calculated for each batch separately
      and take the most common value. The default setting is 'All'.
   -  **ConvertBNToConv**: (Boolean) If True, the BatchNormalization
      operation will be converted to Conv operation. The default is True
      when enable_npu_cnn is True.
   -  **ConvertReduceMeanToGlobalAvgPool**: (Boolean) If True, the
      Reduce Mean operation will be converted to Global Average Pooling
      operation. The default is True when enable_npu_cnn is True.
   -  **SplitLargeKernelPool**: (Boolean) If True, the large kernel
      Global Average Pooling operation will be split into multiple
      Average Pooling operation. The default is True when enable_npu_cnn
      is True.
   -  **ConvertSplitToSlice**: (Boolean) If True, the Split operation
      will be converted to Slice operation. The default is True when
      enable_npu_cnn is True.
   -  **FuseInstanceNorm**: (Boolean) If True, the split instance norm
      operation will be fused to InstanceNorm operation. The default is
      True when enable_npu_cnn is True.
   -  **FuseL2Norm**: (Boolean) If True, a set of L2norm operations will
      be fused to L2Norm operation. The default is True when
      enable_npu_cnn is True.
   -  **FuseLayerNorm**: (Boolean) If True, a set of LayerNorm
      operations will be fused to LayerNorm operation. The default is
      True when enable_npu_cnn is True.
   -  **ConvertClipToRelu**: (Boolean) If True, the Clip operations that
      has a min value of 0 will be converted to ReLU operations. The
      default is True when enable_npu_cnn is True.
   -  **SimulateDPU**: (Boolean) If True, a simulation transformation
      that replaces some operations with an approximate implementation
      will be applied for DPU when enable_npu_cnn is True. The default
      is True.
   -  **ConvertLeakyReluToDPUVersion**: (Boolean) If True, the Leaky
      Relu operation will be converted to DPU version when SimulateDPU
      is True. The default is True.
   -  **ConvertSigmoidToHardSigmoid**: (Boolean) If True, the Sigmoid
      operation will be converted to Hard Sigmoid operation when
      SimulateDPU is True. The default is True.
   -  **ConvertHardSigmoidToDPUVersion**: (Boolean) If True, the Hard
      Sigmoid operation will be converted to DPU version when
      SimulateDPU is True. The default is True.
   -  **ConvertAvgPoolToDPUVersion**: (Boolean) If True, the global or
      kernel-based Average Pooling operation will be converted to DPU
      version when SimulateDPU is True. The default is True.
   -  **ConvertReduceMeanToDPUVersion**: (Boolean) If True, the
      ReduceMean operation will be converted to DPU version when
      SimulateDPU is True. The default is True.
   -  **ConvertSoftmaxToDPUVersion**: (Boolean) If True, the Softmax
      operation will be converted to DPU version when SimulateDPU is
      True. The default is False.
   -  **NPULimitationCheck**: (Boolean) If True, the quantization scale
      will be adjust due to the limitation of DPU/NPU. The default is
      True.
   -  **AdjustShiftCut**: (Boolean) If True, adjust the shift cut of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftBias**: (Boolean) If True, adjust the shift bias of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftRead**: (Boolean) If True, adjust the shift read of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftWrite**: (Boolean) If True, adjust the shift write of
      nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustHardSigmoid**: (Boolean) If True, adjust the pos of hard
      sigmoid nodes when NPULimitationCheck is True. The default is
      True.
   -  **AdjustShiftSwish**: (Boolean) If True, adjust the shift swish
      when NPULimitationCheck is True. The default is True.
   -  **AlignConcat**: (Boolean) If True, adjust the quantization pos of
      concat when NPULimitationCheck is True. The default is True.
   -  **AlignPool**: (Boolean) If True, adjust the quantization pos of
      pooling when NPULimitationCheck is True. The default is True.
   -  **AlignPad**: (Boolean) If True, adjust the quantization pos of
      pad when NPULimitationCheck is True. The default is True.
   -  **AlignSlice**: (Boolean) If True, adjust the quantization pos of
      slice when NPULimitationCheck is True. The default is True.
   -  **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with
      Relu in the model. The default is False.
   -  **CLESteps**: (Int) Specifies the steps for CrossLayerEqualization
      execution when include_cle is set to true, The default is 1, When
      set to -1, an adaptive CrossLayerEqualization will be conducted.
      The default is 1.
   -  **CLETotalLayerDiffThreshold**: (Float) Specifies The threshold
      represents the sum of mean transformations of
      CrossLayerEqualization transformations across all layers when
      utilizing CrossLayerEqualization. The default is 2e-7.
   -  **CLEScaleAppendBias**: (Boolean) Whether the bias be included
      when calculating the scale of the weights, The default is True.
   -  **FastFinetune**: (Dictionary) A parameter used to specify the
      settings for fast finetune.
      
      -  **OptimAlgorithm**: (String) The specified algorithm for fast finetune. Optional values are “adaround” and “adaquant”. The
         “adaround” adjusts the weights rounding function, which is
         relatively stable and might converge faster. The “adaquant” trains
         the weight (and bias optional) directly, so might have a greater
         improvement if the parameters, especially the learning rate and
         batch size, are optimal. The default value is “adaround”.
      -  **OptimDevice**: (String) The compute device for fast finetune.
         Optional values are “cpu”, “hip:0” and “cuda:0”. The default value
         is “cpu”.
      -  **FixedSeed**: (Int) Seed for random data generator, that makes
         the fast finetuned results could be reproduced.
      -  **DataSize**: (Int) Specifies the size of the data used for
         finetuning. Its recommended setting the batch size of the data to
         1 in the data reader to ensure counting the size accurately. It
         uses all the data from the data reader by default.
      -  **BatchSize**: (Int) Batch size for finetuning. The larger batch
         size, usually the better accuracy but the longer training time.
         The default value is 1.
      -  **NumBatches**: (Int) The mini-batches in a iteration. It should
         always be 1. The default value is 1.
      -  **NumIterations**: (Int) The Iterations for finetuning. The more
         iterations, the better accuracy but the longer training time. The
         default value is 1000.
      -  **LearningRate**: (Float) Learning rate of finetuning for all
         layers. It has a significant impact on the accuracy improvement,
         you need to try some learning rates to get a better result for
         your model. The default value is 0.1 for AdaRound and 0.00001 for
         AdaQuant.
      -  **EarlyStop**: (Bool) If average loss of a certain number of
         iterations decreases comparing with the previous one, the training
         of the layer will stop early. It will accelerate the finetuning
         process and avoid overfitting. The default value is False.
      -  **LRAdjust**: (Tuple) Besides the overall learning rate, users
         could set up a scheme to adjust learning rate further according to
         the mean square error (MSE) between the quantized module and
         original float module. Its a tuple contains two members, the
         first one is a threshold of the MSE and the second one is the new
         learning rate. For example, setting as (1.0, 0.2) means using a
         new learning rate 0.2 for the layer whose MSE is bigger than 1.0.
      -  **TargetOpType**: (List) The target operation types to finetune.
         The default value is [Conv, ConvTranspose, Gemm,
         InstanceNormalization].
      -  **SelectiveUpdate**: (Bool) If the end-to-end accuracy does not
         improve after finetuned a certain layer, discard the optimized
         weight (and bias) of the layer. The default value is False.
      -  **UpdateBias**: (Bool) Specifies whether to update bias
         parameters during fine-tuning. Its only available for AdaQuant.
         The default value is False.
      -  **OutputQDQ**: (Bool) Specifies whether include the output
         tensors QDQ pair of the compute nodes for finetuning. The default
         value is False.
      -  **DropRatio**: (Float) Specifies the ratio to drop the input
         data from the float module. It ranges from 0 to 1, 0 represents
         the input data is from the float module fully, 1 represents all
         from quantized module. The default value is 0.5.
      -  **LogPeriod**: (Int) Indicate how many iterations to print the
         log once. The default value is NumIterations/10.
   -  **SmoothAlpha**: (Float) This parameter control how much
      difficulty we want to migrate from activation to weights, The
      default value is 0.5.
   -  **RemoveQDQConvRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and Relu will be removed for DPU. The default is
      True.
   -  **RemoveQDQConvLeakyRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and LeakyRelu will be removed for DPU. The default
      is True.
   -  **RemoveQDQConvPRelu**: (Boolean) If True, the QDQ between
      Conv/Add/Gemm and PRelu will be removed for DPU. The default is
      True.
   -  **RemoveQDQInstanceNorm**: (Boolean) If True, the QDQ between
      InstanceNorm and Relu/LeakyRelu/PRelu will be removed for DPU. The
      default is False.
   -  **FoldBatchNorm**: (Boolean) If True, the BatchNormalization
      operation will be fused with Conv, ConvTranspose or Gemm
      operation. The BatchNormalization operation after Concat operation
      will also be fused, if the all input operations of the Concat
      operation are Conv, ConvTranspose or Gemm operatons.The default is
      True.
   -  **FixShapes**: (String) Set the input_shapes of the quantized
      model to a fixed shape by default if not explicitly specified. The
      example: 'FixShapes':'input_1:[1,224,224,3];input_2:[1,96,96,3]'
   -  **MixedPrecisionTensor**: (Dictionary) A parameter used to specify
      the settings for mixed precision tensors. It is a dictionary where
      the keys are of the VitisQuantType/QuantType enumeration type, and
      the values are lists containing tensors that need to be processed
      using mixed precision.
      Example:"MixedPrecisionTensor":{quark.onnx.VitisQuantType.QBFloat16:['/stem/stem.2/Relu_output_0',
      'onnx::Conv_664', 'onnx::Conv_665']} **Note**:If there is a tensor
      with bias, 'Int32Bias' needs set to False.
   -  **FoldRelu**: (Boolean) If True, the Relu will be fold to Conv
      when use VitisQuantFormat. The default is False.
   -  **CalibDataSize**: (Int) This parameter controls how many data are
      used for calibration. The default to using all the data in the
      calibration dataloader.
   -  **SaveTensorHistFig**: (Boolean) If True, save the tensor
      histogram to the file 'tensor_hist' in the working directory. The
      default is False.
   -  **WeightsOnly**: (Boolean) If True, only quantize weights of the
      model. The default is False.

Table 7. Quantize Types can be selected for different Quantize Formats

+-----------------------+-----------------------+-----------------------+
| quant_format          | quant_type            | comments              |
+=======================+=======================+=======================+
| QuantFormat.QDQ       | QuantType.QUInt8      | Implemented by native |
|                       | QuantType.QInt8       | QuantizeLi            |
|                       |                       | near/DequantizeLinear |
+-----------------------+-----------------------+-----------------------+
| quark.onnx            | QuantType.QUInt8      | Implemented by        |
| .VitisQuantFormat.QDQ | QuantType.QInt8       | customized            |
|                       | quark.onnx.V          | VitisQuantizeLinear/  |
|                       | itisQuantType.QUInt16 | VitisDequantizeLinear |
|                       | quark.onnx.           |                       |
|                       | VitisQuantType.QInt16 |                       |
|                       | quark.onnx.V          |                       |
|                       | itisQuantType.QUInt32 |                       |
|                       | quark.onnx.           |                       |
|                       | VitisQuantType.QInt32 |                       |
|                       | quark.onnx.Vi         |                       |
|                       | tisQuantType.QFloat16 |                       |
|                       | quark.onnx.Vit        |                       |
|                       | isQuantType.QBFloat16 |                       |
+-----------------------+-----------------------+-----------------------+

**Note** : For pure UInt8 or Int8 quantization, we recommend that users
set quant_format to QuantFormat.QDQ as it uses native
QuantizeLinear/DequantizeLinear operations which may have better
compatibility and performance.

.. raw:: html

   <!-- 
   ## License
   Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
