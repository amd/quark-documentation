#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Exporting API for PyTorch."""

import json
import tempfile
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import dataclasses

import torch
import torch.nn as nn
from safetensors.torch import save_file
from quark.torch.quantization.tensor_quantize import FakeQuantize
from quark.torch.export.config.config import ExporterConfig
from quark.torch.export.json_export.builder.llm_model_info_builder import create_llm_model_builder
from quark.torch.export.json_export.builder.native_model_info_builder import NativeModelInfoBuilder
from quark.torch.export.json_export.utils.utils import split_model_info
from quark.torch.export.json_export.converter.ammo_converter import AmmoConverter
from quark.torch.utils.log import ScreenLogger

__all__ = ["ModelExporter"]

logger = ScreenLogger(__name__)


class ModelExporter:
    """
    Provides an API for exporting quantized Pytorch deep learning models.
    This class converts the quantized model to json-safetensors files or onnx graph, and saves to export_dir.

    Args:
        config (ExporterConfig): Configuration object containing settings for exporting.
        export_dir (Union[Path, str]): The target export diretory. This could be a string or a pathlib.Path(string) object.
    """

    def __init__(self, config: ExporterConfig, export_dir: Union[Path, str] = tempfile.gettempdir()) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    def export_model_info(self,
                          model: nn.Module,
                          model_type: str,
                          model_dtype: torch.dtype = torch.float16,
                          export_type: str = "vllm-adopt") -> None:
        """
        This function aims to export json and safetensors files of the quantized Pytorch model.
        The model's network architecture is stored in the json file, and parameters including weight, bias, scale, and zero_point are stored in the safetensors file.

        Parameters:
            model (torch.nn.Module): The quantized model to be exported.
            model_type (str): The type of the model, e.g. gpt2, gptj, llama or gptnext.
            model_dtype (torch.dtype): The weight data type of the quantized model. Default is torch.float16.
            export_type (str): The specific format in which the JSON and safetensors files are stored.
                The choices include 'vllm-adopt' and 'native'. Default is vllm-adopt.
                If set to 'vllm-adopt', the exported files are customized for the VLLM compiler.
                The 'native' configuration is currently for internal testing use.

        Returns:
            None

        **Examples**:

            .. code-block:: python

                export_path = "./output_dir"
                from quark.torch import ModelExporter
                from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
                exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
                exporter.export_model_info(model, model_type, model_dtype, export_type="vllm-adopt")

        Note:
            Since the export_type "native" is only for internal testing use currently, this function is only used to export files required by the VLLM compiler.
            Supported quantization types include fp8, int4_per_group, and w4a8_per_group.
            Supported models include Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.
        """

        params_dict: Dict[str, torch.Tensor] = {}
        if export_type == "native":
            builder = NativeModelInfoBuilder(model=model,
                                             model_type=model_type,
                                             model_dtype=model_dtype,
                                             config=self.config.json_export_config)
            info = builder.build_model_info(params_dict)

        elif export_type == "vllm-adopt":
            llm_builder = create_llm_model_builder(model=model,
                                                   model_type=model_type,
                                                   model_dtype=model_dtype,
                                                   config=self.config.json_export_config)
            model_info = llm_builder.build_model_info()
            info = dataclasses.asdict(model_info)
            split_model_info(info, params_dict)
            converter = AmmoConverter(info, params_dict, config=self.config.json_export_config)
            info = converter.convert()
        else:
            raise ValueError("Only support native and vllm-adopt export type currently")

        json_path = self.export_dir / f"{model_type}.json"
        with open(json_path, "w") as f:
            json.dump(info, f, indent=4)

        # handle tensors shared
        data_ptr_list: List[str] = []
        for key, value in params_dict.items():
            if str(value.data_ptr()) in data_ptr_list:
                params_dict[key] = value.clone()
            else:
                data_ptr_list.append(str(value.data_ptr()))

        params_path = self.export_dir / f"{model_type}.safetensors"
        save_file(params_dict, params_path)

        logger.info("Quantized model exported to {} successfully.".format(self.export_dir))

    def export_onnx_model(self,
                          model: nn.Module,
                          input_args: Union[torch.Tensor, Tuple[float]],
                          input_names: List[str] = [],
                          output_names: List[str] = [],
                          verbose: bool = False,
                          opset_version: Optional[str] = None,
                          do_constant_folding: bool = True,
                          operator_export_type: torch.onnx.OperatorExportTypes = torch.onnx.OperatorExportTypes.ONNX,
                          uint4_int4_flag: bool = False) -> None:
        """
        This function aims to export onnx graph of the quantized Pytorch model.

        Parameters:
            model (torch.nn.Module): The quantized model to be exported.
            input_args (Union[torch.Tensor, Tuple[float]]): Example inputs for this quantized model.
            input_names (List[str]): Names to assign to the input nodes of the onnx graph, in order. Default is empty list.
            output_names (List[str]): Names to assign to the output nodes of the onnx graph, in order. Default is empty list.
            verbose (bool): Flag to control showing verbose log or no. Default is False
            opset_version (Optional[str]): The version of the default (ai.onnx) opset to target.
                If not set, it will be valued the latest version that is stable for the current version of PyTorch.
            do_constant_folding (bool): Apply the constant-folding optimization. Default is False
            operator_export_type (torch.onnx.OperatorExportTypes): Export operator type in onnx graph.
                The choices include OperatorExportTypes.ONNX, OperatorExportTypes.ONNX_FALLTHROUGH, OperatorExportTypes.ONNX_ATEN and OperatorExportTypes.ONNX_ATEN_FALLBACK.
                Default is OperatorExportTypes.ONNX.
            uint4_int4_flag (bool): Flag to indicate uint4/int4 quantized model or not. Default is False.

        Returns:
            None

        **Examples**:

            .. code-block:: python

                from quark.torch import ModelExporter
                from quark.torch.export.config.custom_config import DEFAULT_EXPORTER_CONFIG
                exporter = ModelExporter(config=DEFAULT_EXPORTER_CONFIG, export_dir=export_path)
                exporter.export_onnx_model(model, input_args)

        Note:
            Mix quantization of int4/uint4 and int8/uint8 is not supported currently.
            In other words, if the model contains both quantized nodes of uint4/int4 and uint8/int8, this function cannot be used to export the ONNX graph.
        """
        from quark.torch.export.onnx import convert_model_to_uint4_int4
        for module in model.modules():
            if isinstance(module, FakeQuantize):
                module.disable_observer()
                module.enable_fake_quant()
        onnx_path = str(self.export_dir / "quark_model.onnx")
        torch.onnx.export(model.eval(),
                          input_args,
                          onnx_path,
                          verbose=verbose,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=opset_version,
                          do_constant_folding=do_constant_folding,
                          operator_export_type=operator_export_type)
        if uint4_int4_flag:
            convert_model_to_uint4_int4(onnx_path)

        logger.info("Quantized onnx model exported to {} successfully.".format(onnx_path))
