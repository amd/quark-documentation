Quantization Schemes
====================

Quark for ONNX is capable of handling ``per tensor`` and ``per channel``
quantization, supporting both symmetric and asymmetric methods.

-  **Per Tensor Quantization** means that quantize the tensor with one
   scalar. The scaling factor is a scalar.

-  **Per Channel Quantization** means that for each dimension, typically
   the channel dimension of a tensor, the values in the tensor are
   quantized with different quantization parameters. The scaling factor
   is a 1-D tensor, with the length of the quantization axis. For the
   input tensor with shape ``(D0, ..., Di, ..., Dn)`` and ``ch_axis=i``,
   The scaling factor is a 1-D tensor of length ``Di``.
