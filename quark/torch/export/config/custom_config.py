#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig

# Up_proj gate_proj merge config
UP_GATE_MERGE_CONFIG = JsonExporterConfig(weight_merge_groups=[["*up_proj", "*gate_proj"]])

# default exporter config
DEFAULT_EXPORTER_CONFIG = ExporterConfig(json_export_config=UP_GATE_MERGE_CONFIG,
                                         onnx_export_config=OnnxExporterConfig())

# empty exporter config
EMPTY_EXPORTER_CONFIG = ExporterConfig(json_export_config=JsonExporterConfig(), onnx_export_config=OnnxExporterConfig())
