Quantization Strategies
=======================

Quark for ONNX offers three distinct quantization strategies tailored to
meet the requirements of various HW backends:

-  **Post Training Weight-Only Quantization**: The weights are quantized
   ahead of time but the activations are not quantized(using original
   float data type) during inference.

-  **Post Training Static Quantization**: Post Training Static
   Quantization quantizes both the weights and activations in the model.
   To achieve the best results, this process necessitates calibration
   with a dataset that accurately represents the actual data, which
   allows for precise determination of the optimal quantization
   parameters for activations.

- **Post Training Dynamic Quantization**: Dynamic Quantization quantizes
   the weights ahead of time, while the activations are quantized
   dynamically at runtime. This method allows for a more flexible
   approach, especially when the activation distribution is not
   well-known or varies significantly during inference.

The strategies share the same user API. Users simply need to set the
strategy through the quantization configuration, as demonstrated in the
example above. More details about setting quantization configuration are
in the chapter "Configuring Quark for ONNX"
