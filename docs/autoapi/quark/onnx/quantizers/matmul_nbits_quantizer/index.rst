:orphan:

:py:mod:`quark.onnx.quantizers.matmul_nbits_quantizer`
======================================================

.. py:module:: quark.onnx.quantizers.matmul_nbits_quantizer


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.onnx.quantizers.matmul_nbits_quantizer.MatMulNBitsQuantizer




.. py:class:: MatMulNBitsQuantizer(model: onnx.onnx_pb.ModelProto | str, block_size: int = 128, is_symmetric: bool = False, bits: int = 4, accuracy_level: int | None = None, nodes_to_exclude: Optional[List[str]] = None, algo_config: Optional[WeightOnlyQuantConfig] = None, extra_options: Dict[str, Any] = {})


   Perform 4b quantization of constant MatMul weights


