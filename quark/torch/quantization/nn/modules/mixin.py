#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Optional, Any
from quark.torch.quantization.tensor_quantize import FakeQuantize
from quark.torch.quantization.config.config import QuantizationSpec


class QuantMixin:

    def init_quantizer(self, input_qspec: Optional[QuantizationSpec], output_qspec: Optional[QuantizationSpec],
                       weight_qspec: Optional[QuantizationSpec], **kwargs: Any) -> None:
        self._input_quantizer = FakeQuantize(input_qspec, **kwargs) if input_qspec else None
        self._output_quantizer = FakeQuantize(output_qspec, **kwargs) if output_qspec else None
        self._weight_quantizer = FakeQuantize(weight_qspec, **kwargs) if weight_qspec else None

    @property
    def input_quantizer(self) -> Optional[FakeQuantize]:
        return self._input_quantizer

    @property
    def weight_quantizer(self) -> Optional[FakeQuantize]:
        return self._weight_quantizer

    @property
    def output_quantizer(self) -> Optional[FakeQuantize]:
        return self._output_quantizer
