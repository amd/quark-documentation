:orphan:

:py:mod:`quark.onnx.finetuning.create_torch.create_model_ops`
=============================================================

.. py:module:: quark.onnx.finetuning.create_torch.create_model_ops


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.create_torch.create_model_ops.param_is_symmetric
   quark.onnx.finetuning.create_torch.create_model_ops.extract_padding_params
   quark.onnx.finetuning.create_torch.create_model_ops.extract_padding_params_for_conv
   quark.onnx.finetuning.create_torch.create_model_ops.extract_weight_and_bias
   quark.onnx.finetuning.create_torch.create_model_ops.load_weight_and_bias
   quark.onnx.finetuning.create_torch.create_model_ops.convert_conv
   quark.onnx.finetuning.create_torch.create_model_ops.convert_matmul
   quark.onnx.finetuning.create_torch.create_model_ops.convert_gemm
   quark.onnx.finetuning.create_torch.create_model_ops.convert_norm
   quark.onnx.finetuning.create_torch.create_model_ops.convert_act
   quark.onnx.finetuning.create_torch.create_model_ops.convert_ops_to_modules
   quark.onnx.finetuning.create_torch.create_model_ops.set_modules_original_weight
   quark.onnx.finetuning.create_torch.create_model_ops.get_modules_optimized_weight
   quark.onnx.finetuning.create_torch.create_model_ops.set_modules_original_bias
   quark.onnx.finetuning.create_torch.create_model_ops.get_modules_optimized_bias



.. py:function:: param_is_symmetric(params: List[Any]) -> bool

   Check if parameters are symmetric, all values [2,2,2,2].
   Then we can use only [2,2].


.. py:function:: extract_padding_params(params: List[Any]) -> Any

   Extract padding parameters for Pad layers.


.. py:function:: extract_padding_params_for_conv(params: List[Any]) -> Any

   Padding params in onnx are different than in pytorch. That is why we need to
   check if they are symmetric and cut half or return a padding layer.


.. py:function:: extract_weight_and_bias(params: List[Any]) -> Tuple[numpy.typing.NDArray[Any], Union[numpy.typing.NDArray[Any], None]]

   Extract weights and biases.


.. py:function:: load_weight_and_bias(layer: torch.nn.Module, weight: numpy.typing.NDArray[Any], bias: Union[numpy.typing.NDArray[Any], None]) -> None

   Load weight and bias to a given layer from onnx format.


.. py:function:: convert_conv(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[quark.onnx.finetuning.create_torch.quant_base_ops.QuantizeWrapper, Union[quark.onnx.finetuning.create_torch.quant_base_ops.QuantizeWrapper, None]]

   Use to convert Conv ONNX node to Torch module (or called layer).
      This function supports onnx's Conv and ConvTranspose from 1 to 11.

   :param node : ONNX node.
   :param layer_params : Layer weight and bias parameters.
   :param layer_qinfos : Layer quantization information.
   :return: Converted conv layer, perhaps it has a pad layer.


.. py:function:: convert_matmul(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[quark.onnx.finetuning.create_torch.quant_matmul_ops.QMatMul, None]

   Use to convert MatMul ONNX node to Torch module.
   This function supports onnx's Matmul from 6.
    :param node : ONNX node.
    :param layer_params : Layer weight parameters.
    :param layer_qinfos : Layer quantization informations.
    :return: Converted MatMul layer.


.. py:function:: convert_gemm(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[quark.onnx.finetuning.create_torch.quant_gemm_ops.QGemm, None]

   Use to convert Gemm ONNX node to Torch module.
      This function supports onnx's Instance Norm from 6.

   :param node : ONNX node.
   :param layer_params : Layer weight and bias parameters.
   :param layer_qinfos : Layer quantization information.
   :return: Converted Gemm layer.


.. py:function:: convert_norm(node: onnx.NodeProto, layer_params: List[Any], layer_qinfos: List[Any]) -> Tuple[Union[quark.onnx.finetuning.create_torch.quant_norm_ops.QInstanceNorm2d, quark.onnx.finetuning.create_torch.quant_norm_ops.QLayerNorm], None]

   Use to convert norm (Instance/Layer Norm) ONNX node to Torch module.
      This function supports onnx's Instance Norm from 6.

   :param node : ONNX node.
   :param layer_params : Layer weight and bias parameters.
   :param layer_qinfos : Layer quantization information.
   :return: Converted norm (Instance/Layer Norm) layer.


.. py:function:: convert_act(node: onnx.NodeProto) -> Union[torch.nn.Module, None]

   Use to convert Activation ONNX node to Torch module (or called layer).

   :param node : ONNX node.
   :return: Converted act layer.


.. py:function:: convert_ops_to_modules(onnx_model: onnx.ModelProto) -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[torch.nn.Module], Optional[quark.onnx.finetuning.create_torch.quant_base_ops.QuantizationModule]]

   Convert ONNX operations to Torch modules.


.. py:function:: set_modules_original_weight(module: torch.nn.Module, weight: numpy.typing.NDArray[Any]) -> None

   For setting original float weight 


.. py:function:: get_modules_optimized_weight(module: torch.nn.Module) -> Any

   For getting optimized quantized weight 


.. py:function:: set_modules_original_bias(module: torch.nn.Module, bias: numpy.typing.NDArray[Any]) -> None

   For setting original float bias 


.. py:function:: get_modules_optimized_bias(module: torch.nn.Module) -> Any

   For getting optimized quantized bias 


