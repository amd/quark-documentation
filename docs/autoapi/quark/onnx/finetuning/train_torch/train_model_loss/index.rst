:orphan:

:py:mod:`quark.onnx.finetuning.train_torch.train_model_loss`
============================================================

.. py:module:: quark.onnx.finetuning.train_torch.train_model_loss


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.train_torch.train_model_loss.TrainLoss




.. py:class:: TrainLoss


   Calculates the Reconstruction loss, and Rounding loss
   This class is referenced from the AdaRound algorithm proposed in the following paper:
   "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
   arXiv:2004.10568, 2020."

   .. py:method:: calc_recon_loss(quant_output: torch.Tensor, float_output: torch.Tensor) -> Any
      :staticmethod:

      Calculate Reconstruction Loss using Squared Frobenius Norm
      :param quant_output: Activation output from quantized wrapper module
      :param float_output: Activation output from original float module
      :return: Reconstruction loss


   .. py:method:: calc_round_loss(alpha: torch.Tensor, params: quark.onnx.finetuning.train_torch.train_model_param.TrainParameters, cur_iter: int) -> Any
      :classmethod:

      Calculate Rounding Loss (This is for AdaRound optimization to learn weight rounding)
      :param alpha: Parameter 'alpha' to be optimized
      :param params: Optimization parameters for AdaRound
      :param cur_iter: Current iteration
      :return: Rounding loss



