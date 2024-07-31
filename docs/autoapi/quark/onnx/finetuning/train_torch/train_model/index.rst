:orphan:

:py:mod:`quark.onnx.finetuning.train_torch.train_model`
=======================================================

.. py:module:: quark.onnx.finetuning.train_torch.train_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.train_torch.train_model.ModelOptimizer




.. py:class:: ModelOptimizer


   Optimizes weight or its rounding mode for the quantized wrapper module

   .. py:method:: run(quant_module: torch.nn.Module, inp_data_quant: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any, Any]]], inp_data_float: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any, Any]]], out_data_float: Union[numpy.ndarray[Any, Any], List[numpy.ndarray[Any, Any]]], params: quark.onnx.finetuning.train_torch.train_model_param.TrainParameters) -> None
      :classmethod:

      Run the optimization for the target module
      :param quant_module: Quantized wrapper module which consists of a compute module and a optional act module
      :param inp_data_quant: Quantized wrapper module's input data from all dataset, single array or array list
      :param inp_data_float: Original float module's input data from all dataset, single array or array list
      :param out_data_float: Original float module's output data from all dataset, single array or array list
      :param params: Optimization parameters



