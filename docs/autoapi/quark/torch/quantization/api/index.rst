:py:mod:`quark.torch.quantization.api`
======================================

.. py:module:: quark.torch.quantization.api

.. autoapi-nested-parse::

   Quark Quantization API for PyTorch.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   quark.torch.quantization.api.ModelQuantizer



Functions
~~~~~~~~~

.. autoapisummary::

   quark.torch.quantization.api.load_params



.. py:class:: ModelQuantizer(config: quark.torch.quantization.config.config.Config)


   Provides an API for quantizing deep learning models using PyTorch. This class handles the configuration and processing of the model for quantization based on user-defined parameters. It is essential to ensure that the 'config' provided has all necessary quantization parameters defined. This class assumes that the model is compatible with the quantization settings specified in 'config'.

   Args:
       config (Config): Configuration object containing settings for quantization.


   .. py:method:: quantize_model(model: torch.nn.Module, dataloader: Optional[Union[torch.utils.data.DataLoader[torch.Tensor], torch.utils.data.DataLoader[List[Dict[str, torch.Tensor]]], torch.utils.data.DataLoader[Dict[str, torch.Tensor]], torch.utils.data.DataLoader[List[transformers.feature_extraction_utils.BatchFeature]]]] = None) -> torch.nn.Module

      This function aims to quantize the given PyTorch model to optimize its performance and reduce its size. This function accepts a model and a torch dataloader. The dataloader is used to provide data necessary for calibration during the quantization process. Depending on the type of data provided (either tensors directly or structured as lists or dictionaries of tensors), the function will adapt the quantization approach accordingly.It's important that the model and dataloader are compatible in terms of the data they expect and produce. Misalignment in data handling between the model and the dataloader can lead to errors during the quantization process.

      Parameters:
          model (nn.Module): The PyTorch model to be quantized. This model should be already trained and ready for quantization.
          dataloader (Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]):
              The DataLoader providing data that the quantization process will use for calibration. This can be a simple DataLoader returning
              tensors, or a more complex structure returning either a list of dictionaries or a dictionary of tensors.

      Returns:
          nn.Module: The quantized version of the input model. This model is now optimized for inference with reduced size and potentially improved
          performance on targeted devices.

      **Examples**:

          .. code-block:: python

              # Model & Data preparation
              from transformers import AutoModelForCausalLM, AutoTokenizer
              model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
              model.eval()
              tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
              from quark.torch.quantization.config.config import Config
              from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
              from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver
              DEFAULT_UINT4_PER_GROUP_ASYM_SPEC = QuantizationSpec(dtype=Dtype.uint4,
                                                          observer_cls=PerGroupMinMaxObserver,
                                                          symmetric=False,
                                                          scale_type=ScaleType.float,
                                                          round_method=RoundType.half_even,
                                                          qscheme=QSchemeType.per_group,
                                                          ch_axis=1,
                                                          is_dynamic=False,
                                                          group_size=128)
              DEFAULT_W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=DEFAULT_UINT4_PER_GROUP_ASYM_SPEC)
              quant_config = Config(global_quant_config=DEFAULT_W_UINT4_PER_GROUP_CONFIG)
              from torch.utils.data import DataLoader
              text = "Hello, how are you?"
              tokenized_outputs = tokenizer(text, return_tensors="pt")
              calib_dataloader = DataLoader(tokenized_outputs['input_ids'])

              from quark.torch import ModelQuantizer
              quantizer = ModelQuantizer(quant_config)
              quant_model = quantizer.quantize(model, calib_dataloader)



   .. py:method:: freeze(model: Union[torch.nn.Module, torch.fx.GraphModule]) -> Union[torch.nn.Module, torch.fx.GraphModule]
      :staticmethod:

      Freezes the quantized model by replacing FakeQuantize modules with FreezedFakeQuantize modules.
      If Users want to export quantized model to torch_compile, please freeze model first.

      Args:
          model (nn.Module): The neural network model containing quantized layers.

      Returns:
          nn.Module: The modified model with FakeQuantize modules replaced by FreezedFakeQuantize modules.



.. py:function:: load_params(model: Optional[torch.nn.Module] = None, json_path: str = '', safetensors_path: str = '', pth_path: str = '', quant_mode: quark.torch.quantization.config.type.QuantizationMode = QuantizationMode.eager_mode, compressed: bool = False, reorder: bool = True) -> torch.nn.Module

   Instantiate a quantized model from saved model files, which is generated from "save_params" function.

   Parameters:
       model (torch.nn.Module): The original Pytorch model.
       json_path (str): The path of the saved json file. Only available for eager mode quantization.
       safetensors_path (str): The path of the saved safetensors file. Only available for eager mode quantization.
       pth_path (str): The path of the saved pth file. Only available for fx_graph mode quantization.
       quant_mode (QuantizationMode): The quantization mode. The choice includes "QuantizationMode.eager_mode" and "QuantizationMode.fx_graph_mode". Default is "QuantizationMode.eager_mode".

   Returns:
       nn.Module: The reloaded quantized version of the input model.

   **Examples**:

       .. code-block:: python

           # eager mode:
           from quark.torch import load_params
           model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

       .. code-block:: python

           # fx_graph mode:
           from quark.torch.quantization.api import load_params
           model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)

   Note:
       This function does not support dynamic quantization for now.


