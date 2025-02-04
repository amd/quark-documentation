:py:mod:`quark.torch.pruning.config`
====================================

.. py:module:: quark.torch.pruning.config

.. autoapi-nested-parse::

   Quark Pruning Config API for PyTorch



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.pruning.config.ConfigBase
   quark.torch.pruning.config.Config
   quark.torch.pruning.config.AlgoConfigBase
   quark.torch.pruning.config.AlgoConfig
   quark.torch.pruning.config.OSSCARConfig
   quark.torch.pruning.config.BlockwiseTuningConfig




.. py:class:: ConfigBase




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: Config




   A class that encapsulates comprehensive pruning configurations for a machine learning model, allowing for detailed and hierarchical control over pruning parameters across different model components.

   :param Optional[AlgoConfig] algo_config: Optional configuration for the pruning algorithm, such as OSSCAR. After this process, the params will be reduced. Default is None.
   :param Optional[int] log_severity_level: 0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is 1.


.. py:class:: AlgoConfigBase




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: AlgoConfig




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: OSSCARConfig




   Helper class that provides a standard way to create an ABC using
   inheritance.


.. py:class:: BlockwiseTuningConfig




   Helper class that provides a standard way to create an ABC using
   inheritance.


