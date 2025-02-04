:py:mod:`quark.torch.pruning.api`
=================================

.. py:module:: quark.torch.pruning.api

.. autoapi-nested-parse::

   Quark Peuning API for PyTorch.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.pruning.api.ModelPruner




.. py:class:: ModelPruner(config: quark.torch.pruning.config.Config)


   Provides an API for pruning deep learning models using PyTorch. This class handles the configuration and processing of the model for pruning based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary pruning parameters defined. This class assumes that the model is compatible with the pruning settings specified in 'config'.

   Args:
       config (Config): Configuration object containing settings for pruning.


   .. py:method:: pruning_model(model: torch.nn.Module, dataloader: Optional[Union[torch.utils.data.DataLoader[torch.Tensor], torch.utils.data.DataLoader[List[Dict[str, torch.Tensor]]], torch.utils.data.DataLoader[Dict[str, torch.Tensor]]]] = None) -> torch.nn.Module

      This function aims to prune the given PyTorch model to optimize its performance and reduce its size. This function accepts a model and a torch dataloader. The dataloader is used to provide data necessary for calibration during the pruning process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the pruning approach accordingly.It's important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the pruning process.

      Parameters:
          model (nn.Module): The PyTorch model to be pruning. This model should be already trained and ready for pruning.
          dataloader (Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]):
              The DataLoader providing data that the pruning process will use for calibration. This can be a simple DataLoader returning
              tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

      Returns:
          nn.Module: The pruned version of the input model. This model is now optimized for inference with reduced size and potentially improved
          performance on targeted devices.




