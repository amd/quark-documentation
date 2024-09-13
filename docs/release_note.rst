Release Note
============

New Features (Version 0.5.0)
----------------------------

-  **Quark for PyTorch**

   -  Model Support:

      -  Provided more examples of LLM models quantization:

         -  INT/OCP_FP8E4M3: Llama-3.1, gpt-j-6b, Qwen1.5-MoE-A2.7B, phi-2, Phi-3-mini, Phi-3.5-mini-instruct, Mistral-7B-v0.1
         -  OCP_FP8E4M3: mistralai/Mixtral-8x7B-v0.1, hpcai-tech/grok-1, CohereForAI/c4ai-command-r-plus-08-2024, CohereForAI/c4ai-command-r-08-2024, CohereForAI/c4ai-command-r-plus, CohereForAI/c4ai-command-r-v01, databricks/dbrx-instruct, deepseek-ai/deepseek-moe-16b-chat

      -  Provided more examples of diffusion model quantization: 

         -  Supported models: SDXL, SDXL-Turbo, SD1.5, Controlnet-Canny-SDXL, Controlnet-Depth-SDXL, Controlnet-Canny-SD1.5
         -  Supported schemes: FP8, W8, W8A8 with and without SmoothQuant

   -  PyTorch Quantizer Enhancements:

      -  Supported more CNN models for graph mode quantization.

   -  Data Types:

      -  Supported BFP16, MXFP8_E5M2.
      -  Supported MX6 and MX9. (experimental)

   -  Advanced Quantization Algorithms:

      -  Supported Rotation for Llama models.
      -  Supported SmoothQuant and AWQ for models with GQA and MQA (e.g., LLaMA-3-8B, QWen2-7B).
      -  Provided scripts for generating AWQ configuration automatically.(experimental)
      -  Supported trained quantization thresholds (TQT) and learned step size quantization (LSQ) for better QAT results. (experimental)

   -  Export Capabilities:

      -  Supported reloading function of Json-Safetensors export format.
      -  Enhanced quantization configuration in Json-Safetensors export format.

-  **Quark for ONNX**

   -  ONNX Quantizer Enhancements:

      -  Supported compatibility with onnxruntime version 1.18.
      -  Enhanced quantization support for LLM models.

   -  Quantization Strategy:

      -  Supported dynamic quantization.

   -  Custom operations:

      -  Optimized "BFPFixNeuron" to support running on GPU.

   -  Advanced Quantization Algorithms:

      -  Improved AdaQuant to support BFP data types.

New Features (Version 0.2.0)
----------------------------

-  **Quark for PyTorch**

   -  **PyTorch Quantizer Enhancements**:

      -  Post Training Quantization (PTQ) and Quantization-Aware Training (QAT) are now supported in FX graph mode.
      -  Introduced quantization support of the following modules: torch.nn.Conv2d.

   -  **Data Types**:

      -  `OCP Microscaling (MX) is supported. Valid element data types include INT8, FP8_E4M3, FP4, FP6_E3M2, and FP6_E2M3. <./pytorch/tutorial_mx.html>`__

   -  **Export Capabilities**:

      -  `Quantized models can now be exported in GGUF format. The exported GGUF model is runnable with llama.cpp. Only Llama2 is supported for now. <./pytorch/tutorial_gguf.html>`__
      -  Introduced Quark's native Json-Safetensors export format, which is identical to AutoFP8 and AutoAWQ when used for FP8 and AWQ quantization.

   -  **Model Support**:

      -  Added support for SDXL model quantization in eager mode, including fp8 per-channel and per-tensor quantization.
      -  Added support for PTQ and QAT of CNN models in graph mode, including architectures like ResNet.

   -  **Integration with other toolkits**:

      -  Provided the integrated example with APL(AMD Pytorch-light,internal project name), supporting the invocation of APL's INT-K, BFP16, and BRECQ.
      -  Introduced the experimental Quark extension interface, enabling seamless integration of Brevitas for Stable Diffusion and Imagenet classification model quantization.

-  **Quark for ONNX**

   -  **ONNX Quantizer Enhancements**:

      -  Multiple optimization and refinement strategies for different deployment backends.
      -  Supported automatic mixing precision to balance accuracy and performance.

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

      -  "BFPFixNeuron" which supports block floating-point data type. It can run on the CPU on Windows, and on both the CPU and GPU on Linux. 
      -  "VitisQuantizeLinear" and "VitisDequantizeLinear" which support INT32/UINT32, Float16, Bfloat16, INT16/UINT16 quantization.
      -  "VitisInstanceNormalization" and "VitisLSTM" which have customized Bfloat16 kernels.
      -  All custom operations support running on the CPU on both Linux and Windows.

   -  **Advanced Quantization Algorithms**:

      -  Supported CLE, BiasCorrection, AdaQuant, AdaRound and SmoothQuant.

   -  **Operating System Support**:

      -  Linux and Windows.

New Features (Version 0.1.0)
----------------------------

-  **Quark for PyTorch**

   -  **Pytorch Quantizer Enhancements**:

      -  Eager mode is supported.
      -  Post Training Quantization (PTQ) is now available.
      -  Automatic in-place replacement of nn.module operations.
      -  Quantization of the following modules is supported: torch.nn.linear.
      -  The customizable calibration process is introduced.

   -  **Quantization Strategy**:

      -  Symmetric and asymmetric quantization are supported.
      -  Weight-only, dynamic, and static quantization modes are available.

   -  **Quantization Granularity**:

      -  Support for per-tensor, per-channel, and per-group granularity.

   -  **Data Types**:

      -  Multiple data types are supported, including float16, bfloat16, int4, uint4, int8, and fp8 (e4m3fn).

   -  **Calibration Methods**:

      -  MinMax, Percentile, and MSE calibration methods are now supported.

   -  **Large Language Model Support**:

      -  FP8 KV-cache quantization for large language models(LLMs).

   -  **Advanced Quantization Algorithms**:

      -  Support SmoothQuant, AWQ(uint4), and GPTQ(uint4) for LLMs. (Note: AWQ/GPTQ/SmoothQuant algorithms are currently limited to single GPU usage.)

   -  **Export Capabilities**:

      -  Export of Q/DQ quantized models to ONNX and vLLM-adopted JSON-safetensors format now supported.

   -  **Operating System Support**:

      -  Linux (supports ROCM and CUDA)
      -  Windows (support CPU only).
