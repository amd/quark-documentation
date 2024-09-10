:py:mod:`quark.torch.extensions.brevitas.verification`
======================================================

.. py:module:: quark.torch.extensions.brevitas.verification

.. autoapi-nested-parse::

   Config verificiation helper functions for Brevitas quantizer.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.extensions.brevitas.verification.ConfigVerifier




.. py:class:: ConfigVerifier


   This is a helper utility to inspect Brevitas quantization configs and ensure they are valid. It'll warn the user about parameters that need to be set or that won't have any effect and it will highlight possible improvements where possible.


