:py:mod:`quark.onnx.tools.convert_resize_fs_to_pof2s`
=====================================================

.. py:module:: quark.onnx.tools.convert_resize_fs_to_pof2s

.. autoapi-nested-parse::

   Convert resize op's float scale to pof2s.

       Example : python -m quark.onnx.tools.convert_resize_fs_to_pof2s --input_model INPUT_MODEL_PATH --output_model OUTPUT_MODEL_PATH



Module Contents
---------------


Functions
~~~~~~~~~

.. autoapisummary::

   quark.onnx.tools.convert_resize_fs_to_pof2s.scale2pos
   quark.onnx.tools.convert_resize_fs_to_pof2s.pos2scale



.. py:function:: scale2pos(scale: float) -> int

   Obtain the fixed-point position corresponding to the scale.
   To avoid generating infinity during computations,
   the range of scale is limited.
   :param scale: the scale
   :return: the fixed-point position


.. py:function:: pos2scale(pos: int) -> float

   Obtain the scale corresponding to the fixed-point position.
   :param scale: the fixed-point position
   :return: the scale


