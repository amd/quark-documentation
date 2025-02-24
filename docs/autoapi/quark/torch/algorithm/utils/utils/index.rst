:orphan:

:py:mod:`quark.torch.algorithm.utils.utils`
===========================================

.. py:module:: quark.torch.algorithm.utils.utils


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.algorithm.utils.utils.TensorData




.. py:class:: TensorData(data: List[torch.Tensor], targets: List[torch.Tensor], device: torch.device)




   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


