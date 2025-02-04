:py:mod:`quark.onnx.tools.save_tensor_hist`
===========================================

.. py:module:: quark.onnx.tools.save_tensor_hist

.. autoapi-nested-parse::

   A tool for showing the activation distribution of a model.

       Example : python -m quark.onnx.tools.save_tensor_hist --input_model [INPUT_MODEL_PATH] --data_path [CALIB_DATA_PATH]  --output_path [OUTPUT_PATH]



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.tools.save_tensor_hist.HistDataReader




.. py:class:: HistDataReader(model_path: str, data_path: str, input_shape: Dict[str, List[int]] = {})




   A CalibrationDataReader using random data for rapid quantiation.

   .. py:method:: get_next() -> Optional[Dict[str, numpy.typing.NDArray[numpy.float32]]]

      Get next feed data
      :return: feed dict for the model



