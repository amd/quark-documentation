:orphan:

:py:mod:`quark.onnx.finetuning.torch_utils`
===========================================

.. py:module:: quark.onnx.finetuning.torch_utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.torch_utils.CachedDataset



Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.torch_utils.setup_seed
   quark.onnx.finetuning.torch_utils.convert_onnx_to_torch
   quark.onnx.finetuning.torch_utils.convert_torch_to_onnx
   quark.onnx.finetuning.torch_utils.save_torch_model
   quark.onnx.finetuning.torch_utils.parse_options_to_params
   quark.onnx.finetuning.torch_utils.train_torch_module_api
   quark.onnx.finetuning.torch_utils.optimize_module



.. py:function:: setup_seed(seed: int) -> None

   Set the seed for random functions


.. py:function:: convert_onnx_to_torch(onnx_model: onnx.ModelProto, float_weight: Optional[numpy.typing.NDArray[Any]] = None, float_bias: Optional[numpy.typing.NDArray[Any]] = None) -> torch.nn.Module

   Convert a onnx model to torch module. Since the onnx model is always a quantized one,
   which has a folded QuantizeLinear in the weight tensor's QDQ.
   In order to obtain the original float weight without loss for the quantize wrapper,
   an additional float weight needs to be feed in.
   :param onnx_model: instance of onnx model
   :param float_weight: float weight
   :param float_bias: float bias
   :return: a torch nn.Module instance


.. py:function:: convert_torch_to_onnx(torch_model: torch.nn.Module, input_data: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]]) -> onnx.ModelProto

   Convert a torch model to onnx model, do not support models bigger than 2GB
   :param torch_model: instance of torch model
   :param input_data: numpy array for single input or list for multiple inputs
   :return: the onnx model instance


.. py:function:: save_torch_model(torch_model: torch.nn.Module, model_path: str, input_data: Union[None, numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]] = None) -> None

   Save a torch model to file
   :param torch_model: instance of torch model
   :param model path: the path to save
   :param input data: the input numpy array data for jit.trace


.. py:class:: CachedDataset(data_reader: onnxruntime.quantization.CalibrationDataReader)




   Cache data from calibration data reader of onnxruntime-based quantizer.


.. py:function:: parse_options_to_params(extra_options: Dict[str, Any]) -> quark.onnx.finetuning.train_torch.train_model_param.TrainParameters

   Get train parameters from extra options


.. py:function:: train_torch_module_api(quant_module: torch.nn.Module, inp_data_quant: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], inp_data_float: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], out_data_float: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], extra_options: Any) -> Any

   Call torch training classes for adaround or adaquant


.. py:function:: optimize_module(quant_model: onnx.ModelProto, float_weight: numpy.typing.NDArray[Any], float_bias: Optional[numpy.typing.NDArray[Any]], inp_data_quant: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], inp_data_float: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], out_data_float: Union[numpy.typing.NDArray[Any], List[numpy.typing.NDArray[Any]]], extra_options: Any) -> Any

   Optimize the onnx module with fast finetune algorithms by torch optimizer


