:orphan:

:py:mod:`quark.torch.export.nn.modules.export_operator`
=======================================================

.. py:module:: quark.torch.export.nn.modules.export_operator


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.export.nn.modules.export_operator.ExportLinear
   quark.torch.export.nn.modules.export_operator.ExportConv2d




.. py:class:: ExportLinear(quant_linear: quark.torch.quantization.nn.modules.quantize_linear.QuantLinear, custom_mode: Optional[str] = None)




   Exporting version of nn.Linear




.. py:class:: ExportConv2d(quant_conv2d: quark.torch.quantization.nn.modules.quantize_conv.QuantConv2d, custom_mode: Optional[str] = None)




   Exporting version of nn.Conv2d




