:orphan:

:py:mod:`quark.onnx.quarot`
===========================

.. py:module:: quark.onnx.quarot


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.quarot.QuaRot




.. py:class:: QuaRot(onnx_model_path: str, input_model: onnx.ModelProto, r_matrixs: Dict[str, numpy.ndarray[Any, Any]], rotation_config_info: Dict[Any, Any], is_large: bool = True)


   A class for quarot
   Args:
       onnx_model_path (str): The ONNX model path to be rotated.
       input_model (onnx.ModelProto): The ONNX model to be rotated.
       r_matrixs (Dict[str, np.ndarray]): The dict of rotation matrix
       rotation_config_info (Dict): The dict to define which sub-structure need rotation.
       is_large (bool): True if the model size is larger than 2GB.

   .. py:method:: rotate_in_channels(data_tensor: numpy.ndarray[Any, Any], rotation_matrix: numpy.ndarray[Any, Any], transpose: bool) -> numpy.ndarray[Any, Any]

      Rotate the input channels of a weight matrix, i.e., inverse transformation to origin field


   .. py:method:: rotate_out_channels(data_tensor: numpy.ndarray[Any, Any], rotation_matrix: numpy.ndarray[Any, Any], transpose: bool) -> numpy.ndarray[Any, Any]

      Rotate the output channels of a weight matrix, i.e., transformation to orthogonal field



