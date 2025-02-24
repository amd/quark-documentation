:orphan:

:py:mod:`quark.torch.quantization.observer.tqt_observer`
========================================================

.. py:module:: quark.torch.quantization.observer.tqt_observer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.observer.tqt_observer.TQTObserver




.. py:class:: TQTObserver(qspec: quark.torch.quantization.config.config.QuantizationSpec, device: Optional[torch.device] = None)




   Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.


   .. py:method:: get_fix_position() -> int

      (1) TQT: qx = clip(round(fx / scale)) * scale, scale = 2^ceil(log2t) / 2^(b-1)
      (2) NndctFixNeron: qx = clip(round(fx * scale)) * (1 / scale), scale = 2^fp
      Let (1) equals (2), we can get
      (3): 2^(b-1) / 2^ceil(log2t) = 2^fp
       => fp = b - 1 - ceil(log2t)

      For more details, see nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device



