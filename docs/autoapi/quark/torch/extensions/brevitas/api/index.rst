:py:mod:`quark.torch.extensions.brevitas.api`
=============================================

.. py:module:: quark.torch.extensions.brevitas.api

.. autoapi-nested-parse::

   Quark Quantization API for Brevitas.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.extensions.brevitas.api.ModelQuantizer
   quark.torch.extensions.brevitas.api.ModelExporter




.. py:class:: ModelQuantizer(config: quark.torch.extensions.brevitas.config.Config)


   Provides an API for quantizing deep learning models using Brevitas.

   The way this class interacts with Brevitas is based on the brevitas ptq example found here:
   https://github.com/Xilinx/brevitas/tree/master/src/brevitas_examples/imagenet_classification/ptq

   Example usage:
       weight_spec = QuantizationSpec()
       global_config = QuantizationConfig(weight=weight_spec)
       config = Config(global_quant_config=global_config)
       quantizer = ModelQuantizer(config)
       quant_model = quantizer.quantize_model(model, calib_dataloader)

   .. py:method:: quantize_model(model: torch.nn.Module, calib_loader: Optional[torch.utils.data.DataLoader] = None) -> torch.nn.Module

      Quantizes the given model.

      - `model`: The model to be quantized.
      - `calib_loader`: A dataloader for calibration data, technically optional but required for most quantization processes.



.. py:class:: ModelExporter(export_path: str)


   Provides an API for exporting pytorch models quantized with Brevitas.
   This class converts the quantized model to an onnx graph, and saves it to the specified export_path.

   Example usage:
       exporter = ModelExporter("model.onnx")
       exporter.export_onnx_model(quant_model, args=torch.ones(1, 1, 784))

   .. py:method:: export_onnx_model(model: torch.nn.Module, args: Union[torch.Tensor, Tuple[torch.Tensor]]) -> None

      Exports a model to onnx.

      - `model`: The pytorch model to export.
      - `args`: Representative tensor(s) in the same shape as the expected input(s) (can be zero, random, ones or even real data).



