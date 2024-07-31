:orphan:

:py:mod:`quark.onnx.finetuning.train_torch.train_model_param`
=============================================================

.. py:module:: quark.onnx.finetuning.train_torch.train_model_param


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.finetuning.train_torch.train_model_param.TrainParameters




.. py:class:: TrainParameters(data_loader: Union[torch.utils.data.DataLoader[Any], None] = None, data_size: Union[int, None] = None, fixed_seed: Union[int, None] = None, num_batches: int = 1, num_iterations: int = 1000, batch_size: int = 1, initial_lr: float = 0.1, optim_algo: str = 'adaround', optim_device: str = 'cpu', lr_adjust: Any = (), selective_update: bool = False, early_stop: bool = False, log_period: Union[float, int] = 100, update_bias: bool = True, reg_param: float = 0.01, beta_range: Tuple[int, int] = (20, 2), warm_start: float = 0.2, drop_ratio: float = 0.5, block_recon: bool = False, dummy_path: str = '')


   Configuration parameters for AdaRound and AdaQuant algorithms.

   The AdaRound is referenced from the following paper:
   "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
   arXiv:2004.10568, 2020."

   The AdaQuant is referenced from the following paper:
   "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming,
   arXiv:2006.10518, 2020."


