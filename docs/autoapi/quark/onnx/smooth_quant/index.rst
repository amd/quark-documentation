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
   Args:
       onnx_model_path (str): The ONNX model path to be smoothed.
       input_model (onnx.ModelProto): The ONNX model to be smoothed.
       dataloader (torch.utils.data.DataLoader): The dataloader used for calibrate.
       alpha (float): The extent to which the difficulty of quantification is shifted from activation to weighting.
       is_large (bool): True if the model size is larger than 2GB.


