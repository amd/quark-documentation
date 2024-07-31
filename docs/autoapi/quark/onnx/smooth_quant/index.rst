:orphan:

:py:mod:`quark.onnx.smooth_quant`
=================================

.. py:module:: quark.onnx.smooth_quant


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.smooth_quant.SmoothQuant




.. py:class:: SmoothQuant(onnx_model_path: str, input_model: onnx.ModelProto, dataloader: torch.utils.data.DataLoader, alpha: float, is_large: bool = True, providers: List[str] = ['CPUExecutionProvider'])


   A class for model smooth
   :param onnx_model_path: The ONNX model path to be smoothed.
   :type onnx_model_path: str
   :param input_model: The ONNX model to be smoothed.
   :type input_model: onnx.ModelProto
   :param dataloader: The dataloader used for calibrate.
   :type dataloader: torch.utils.data.DataLoader
   :param alpha: The extent to which the difficulty of quantification is shifted from activation to weighting.
   :type alpha: float
   :param is_large: True if the model size is larger than 2GB.
   :type is_large: bool


