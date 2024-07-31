:orphan:

:py:mod:`quark.onnx.finetuning.onnx_evaluate`
=============================================

.. py:module:: quark.onnx.finetuning.onnx_evaluate


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.onnx_evaluate.create_session
   quark.onnx.finetuning.onnx_evaluate.inference_model
   quark.onnx.finetuning.onnx_evaluate.average_L2



.. py:function:: create_session(onnx_model: Union[onnx.ModelProto, str]) -> onnxruntime.InferenceSession

   Create a inference session for the onnx model and register libraries for it.
   :param onnx_model: the proto or the path of the onnx model
   :return: the created inference session


.. py:function:: inference_model(onnx_model: Union[onnx.ModelProto, str], data_reader: quark.onnx.quant_utils.CachedDataReader, data_num: Union[int, None] = None, output_index: Union[int, None] = None) -> List[List[numpy.ndarray[Any, Any]]]

   Run the onnx model and feeding it with the data from the cached data reader.
   :param onnx_model: the proto or the path of the onnx model
   :param data_reader: the cached data reader
   :param data_num: how many samples will be used in the data reader
   :param output_index: which output will be chosen to calculate L2
   :return: the results after inference


.. py:function:: average_L2(float_results: List[List[numpy.ndarray[Any, Any]]], quant_results: List[List[numpy.ndarray[Any, Any]]]) -> Any

   Calculate the average L2 distance between the float model and the quantized model.
   :param float_results: the result of the float model
   :param quant_results: the result of the quant model
   :return: the average L2 distance


