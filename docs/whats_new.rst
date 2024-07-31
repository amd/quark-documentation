What’s New
==========

New Features (Version 0.2.0)
----------------------------

-  **Quark for PyTorch**

   -  **PyTorch Quantizer Enhancements**:

      -  Post Training Quantization (PTQ) and Quantization-Aware
         Training (QAT) are now supported in FX graph mode.
      -  Introduced quantization support of the following modules:
         torch.nn.Conv2d.

   -  **Data Types**:

      -  `OCP Microscaling (MX) is supported. Valid element data types
         include INT8, FP8_E4M3, FP4, FP6_E3M2, and
         FP6_E2M3. <./pytorch/tutorial_mx.html>`__

   -  **Export Capabilities**:

      -  `Quantized models can now be exported in GGUF format. The
         exported GGUF model is runnable with llama.cpp. Only Llama2 is
         supported for now. <./pytorch/tutorial_gguf.html>`__
      -  Introduced Quark’s native Json-Safetensors export format, which
         is identical to AutoFP8 and AutoAWQ when used for FP8 and AWQ
         quantization.

   -  **Model Support**:

      -  Added support for SDXL model quantization in eager mode,
         including fp8 per-channel and per-tensor quantization.
      -  Added support for PTQ and QAT of CNN models in graph mode,
         including architectures like ResNet.

   -  **Integration with other toolkits**:

      -  Provided the integrated example with APL(AMD
         Pytorch-light,internal project name), supporting the invocation
         of APL’s INT-K, BFP16, and BRECQ.
      -  Introduced the experimental Quark extension interface, enabling
         seamless integration of Brevitas for Stable Diffusion and
         Imagenet classification model quantization.

-  **Quark for ONNX**

   -  **ONNX Quantizer Enhancements**:

      -  Multiple optimization and refinement strategies for different
         deployment backends.
      -  Supported automatic mixing precision to balance accuracy and
         performance.

   -  **Quantization Strategy**:

      -  Supported symmetric and asymmetric quantization.
      -  Supported float scale, INT16 scale and power-of-two scale.
      -  Supported static quantization and weight-only quantization.

   -  **Quantization Granularity**:

      -  Supported for per-tensor and per-channel granularity.

   -  **Data Types**:

      -  Multiple data types are supported, including INT32/UINT32,
         Float16, Bfloat16, INT16/UINT16, INT8/UINT8 and BFP.

   -  **Calibration Methods**:

      -  MinMax, Entropy and Percentile for float scale.
      -  MinMax for INT16 scale.
      -  NonOverflow and MinMSE for power-of-two scale.

   -  **Custom operations**:

      -  “BFPFixNeuron” which supports block floating-point data type.
      -  “VitisQuantizeLinear” and “VitisDequantizeLinear” which support
         INT32/UINT32, Float16, Bfloat16, INT16/UINT16 quantization.
      -  “VitisInstanceNormalization” and “VitisLSTM” which have
         customized Bfloat16 kernels.
      -  All custom operations only support running on CPU.

   -  **Advanced Quantization Algorithms**:

      -  Supported CLE, BiasCorrection, AdaQuant, AdaRound and
         SmoothQuant.

   -  **Operating System Support**:

      -  Linux and Windows.
