:orphan:

:py:mod:`quark.onnx.quantize`
=============================

.. py:module:: quark.onnx.quantize


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.quantize.quantize_dynamic



.. py:function:: quantize_dynamic(model_input: Union[str, pathlib.Path, onnx.ModelProto], model_output: Union[str, pathlib.Path], op_types_to_quantize: Union[List[str], None] = [], per_channel: bool = False, reduce_range: bool = False, weight_type: onnxruntime.quantization.quant_utils.QuantType = QuantType.QInt8, nodes_to_quantize: List[str] = [], nodes_to_exclude: List[str] = [], subgraphs_to_exclude: List[Tuple[List[str]]] = [], use_external_data_format: bool = False, debug_mode: bool = False, extra_options: Optional[Dict[str, Any]] = {}) -> None

   Given an onnx model, create a quantized onnx model and save it into a file

   Args:
       model_input: file path of model or ModelProto to quantize
       model_output: file path of quantized model
       op_types_to_quantize:
           specify the types of operators to quantize, like ['Conv'] to quantize Conv only.
           It quantizes all supported operators by default.
       per_channel: quantize weights per channel
       reduce_range:
           quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine,
           especially for per-channel mode
       weight_type:
           quantization data type of weight. Please refer to
           https://onnxruntime.ai/docs/performance/quantization.html for more details on data type selection
       nodes_to_quantize:
           List of nodes names to quantize. When this list is not None only the nodes in this list
           are quantized.
           example:
           [
               'Conv__224',
               'Conv__252'
           ]
       nodes_to_exclude:
           List of nodes names to exclude. The nodes in this list will be excluded from quantization
           when it is not None.
       subgraphs_to_exclude:
           List of start and end nodes names of subgraphs to exclude. The nodes matched by the subgraphs will be excluded from quantization
           when it is not None.
       use_external_data_format: option used for large size (>2GB) model. Set to False by default.
       extra_options:
           key value pair dictionary for various options in different case. Current used:
               extra.Sigmoid.nnapi = True/False  (Default is False)
               ActivationSymmetric = True/False: symmetrize calibration data for activations (default is False).
               WeightSymmetric = True/False: symmetrize calibration data for weights (default is True).
               EnableSubgraph = True/False :
                   Default is False. If enabled, subgraph will be quantized. Dynamic mode currently is supported. Will
                   support more in the future.
               ForceQuantizeNoInputCheck = True/False :
                   By default, some latent operators like maxpool, transpose, do not quantize if their input is not
                   quantized already. Setting to True to force such operator always quantize input and so generate
                   quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.
               MatMulConstBOnly = True/False:
                   Default is True for dynamic mode. If enabled, only MatMul with const B will be quantized.


