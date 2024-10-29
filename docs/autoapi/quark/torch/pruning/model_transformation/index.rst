:orphan:

:py:mod:`quark.torch.pruning.model_transformation`
==================================================

.. py:module:: quark.torch.pruning.model_transformation


Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.pruning.model_transformation.prune_weights_tool



.. py:function:: prune_weights_tool(weights: torch.Tensor, bias: torch.Tensor, mask: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]

   Weight pruning tool that removes weights based on the given mask.

   Parameters:
   weights: np.ndarray - The weight tensor
   bias: np.ndarray - The bias tensor
   mask: List[bool] - A boolean mask indicating which weights to keep

   Returns:
   np.ndarray - The pruned weight tensor


