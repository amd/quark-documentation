:orphan:

:py:mod:`quark.torch.algorithm.processor`
=========================================

.. py:module:: quark.torch.algorithm.processor


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.processor.BaseAlgoProcessor




.. py:class:: BaseAlgoProcessor(model: torch.nn.Module, quant_algo_config: Any, calib_data: Union[torch.utils.data.DataLoader[torch.Tensor], torch.utils.data.DataLoader[List[Dict[str, torch.Tensor]]], torch.utils.data.DataLoader[Dict[str, torch.Tensor]]])




   Helper class that provides a standard way to create an ABC using
   inheritance.


