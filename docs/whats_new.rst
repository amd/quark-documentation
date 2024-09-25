What's New
==========

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
