#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import quark.torch.kernel  # noqa

from typing import Optional, List, Dict, Any, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from quark.torch.quantization.observer.observer import ObserverBase, PlaceholderObserver, UniformScalingObserver, PerTensorMinMaxObserver, PerTensorMSEObserver, PerTensorPercentileObserver
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType
from quark.torch.kernel import RoundMode  # type: ignore [attr-defined]
from quark.torch.quantization.utils import calculate_qmin_qmax


class FakeQuantizeBase(ABC, nn.Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self) -> None:
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()
        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def enable_fake_quant(self, enabled: bool = True) -> None:

        self.fake_quant_enabled[0] = 1 if enabled else 0

    def disable_fake_quant(self) -> None:
        self.enable_fake_quant(False)

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    def disable_observer(self) -> None:
        self.enable_observer(False)

    @property
    def is_enabled(self) -> bool:
        return self.observer_enabled[0].item() == 1 or self.fake_quant_enabled[0].item() == 1


def _get_num_bits(dtype: Dtype) -> Union[int, Tuple[int, int]]:
    if dtype in [Dtype.int4, Dtype.uint4]:
        return 4
    elif dtype == Dtype.int8:
        return 8
    elif dtype == Dtype.fp8_e4m3:
        return (4, 3)
    else:
        raise TypeError()


class FakeQuantize(FakeQuantizeBase):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        quant_spec: QuantizationSpec,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs.get("reload", False) is True:
            self.fake_quant_enabled[0] = 1
            self.observer_enabled[0] = 0

        # Set properties with Quant Config
        self.dtype = quant_spec.dtype
        self.is_dynamic = quant_spec.is_dynamic
        self.qscheme = quant_spec.qscheme
        self.ch_axis = quant_spec.ch_axis
        self.group_size = quant_spec.group_size
        self.symmetric = quant_spec.symmetric
        self.round_method = quant_spec.round_method
        self.scale_type = quant_spec.scale_type

        self.device: Optional[torch.device] = None

        if self.observer_enabled[0] == 1:
            # Populate observer_kwargs
            self.observer = self.create_observer(quant_spec)

            if quant_spec.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:
                # TODO: keeping self.quant_min/max for BC; remove after a couple releases
                # Users should use self.observer.quant_min
                assert isinstance(self.observer, (UniformScalingObserver))
                self.quant_min = self.observer.quant_min
                self.quant_max = self.observer.quant_max
                self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
                self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
                assert quant_spec.round_method
                self.round_method = quant_spec.round_method
                assert self.round_method == RoundType.half_even
                self._num_bits = _get_num_bits(quant_spec.dtype)
            elif quant_spec.dtype == Dtype.fp8_e4m3:
                assert isinstance(self.observer,
                                  (PerTensorMinMaxObserver, PerTensorMSEObserver, PerTensorPercentileObserver))
                self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
                self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
                self.register_buffer('amax', torch.tensor([0.0], dtype=torch.float))
                self.maxbound = 448.0
                self._num_bits = _get_num_bits(quant_spec.dtype)
        else:
            if quant_spec.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:
                # TODO: keeping self.quant_min/max for BC; remove after a couple releases
                # Users should use self.observer.quant_min
                self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
                self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
                assert quant_spec.round_method
                self.round_method = quant_spec.round_method
                assert self.round_method == RoundType.half_even
                self._num_bits = _get_num_bits(quant_spec.dtype)
                self.quant_min, self.quant_max = calculate_qmin_qmax(quant_spec.dtype)
            elif quant_spec.dtype == Dtype.fp8_e4m3:
                self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
                self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
                self.register_buffer('amax', torch.tensor([0.0], dtype=torch.float))
                self.maxbound = 448.0
                self._num_bits = _get_num_bits(quant_spec.dtype)

    @staticmethod
    def create_observer(quant_spec: QuantizationSpec) -> ObserverBase:
        if quant_spec.observer_cls is not None:
            return quant_spec.observer_cls(quant_spec)
        else:
            if quant_spec.dtype in [Dtype.bfloat16, Dtype.float16]:
                return PlaceholderObserver(quant_spec)
            else:
                raise ValueError(f'observer_cls cannot be None when dtype is {quant_spec.dtype.name}')

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.observer, (UniformScalingObserver))
        return self.observer._calculate_qparams()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.device = X.device
        self.input_origin_type = X.dtype

        if self.dtype in [Dtype.bfloat16, Dtype.float16]:
            quant_dtype = {Dtype.bfloat16: torch.bfloat16, Dtype.float16: torch.float16}.get(self.dtype)
            if self.input_origin_type != quant_dtype:
                X = X.to(quant_dtype)
                X = X.to(self.input_origin_type)
        elif self.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:

            # Reshape for per_group input. TODO: Write a new kernel named fake_quantize_per_group_affine
            if self.qscheme in [QSchemeType.per_group] and (self.observer_enabled[0] == 1
                                                            or self.fake_quant_enabled[0] == 1):
                org_x_shape = X.shape
                if self.group_size and self.group_size > 0:
                    assert org_x_shape[-1] % self.group_size == 0
                    X = X.reshape(-1, self.group_size)
                assert X.dim() == 2
                self.scale = self.scale.reshape(-1)
                self.zero_point = self.zero_point.reshape(-1)

            # Do observation
            if self.observer_enabled[0] == 1 or self.is_dynamic:
                if self.is_dynamic:
                    self.observer.reset_min_max_vals()
                self.observer(X.detach())
                _scale, _zero_point = self._calculate_qparams()
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)
                self.scale = self.scale.to(self.device).to(_scale.dtype)
                self.zero_point = self.zero_point.to(self.device).to(_zero_point.dtype)

            # Do fake quantize
            if self.fake_quant_enabled[0] == 1:
                if self.qscheme == QSchemeType.per_tensor:
                    X = quark.torch.kernel.fake_quantize_per_tensor_affine(  # type: ignore[attr-defined]
                        X, self.scale, self.zero_point.to(torch.int), self.quant_min, self.quant_max,
                        RoundMode.NEARBYINT)
                elif self.qscheme in [QSchemeType.per_channel, QSchemeType.per_group]:
                    assert self.ch_axis is not None
                    X = quark.torch.kernel.fake_quantize_per_channel_affine(  # type: ignore[attr-defined]
                        X, self.scale, self.zero_point.to(torch.int), self.ch_axis, self.quant_min, self.quant_max,
                        RoundMode.NEARBYINT)
                else:
                    ValueError(f"Do not support QSchema as {self.qscheme}")

            # Reshape back for per_group input. TODO: Remove this and write a new kernel named fake_quantize_per_group_affine
            if self.qscheme in [QSchemeType.per_group] and ((self.observer_enabled[0] == 1) or
                                                            (self.fake_quant_enabled[0] == 1)):
                X = X.reshape(org_x_shape)
                self.scale = self.scale.reshape(X.shape[0], -1)
                self.zero_point = self.zero_point.reshape(X.shape[0], -1)

        elif self.dtype == Dtype.fp8_e4m3:

            # Do observation
            if self.observer_enabled[0] == 1:
                self.observer(X.detach())
                _scale, _zero_point = self._calculate_qparams()
                if self.scale.shape != _scale.shape:
                    self.scale.resize_(_scale.shape)
                    self.zero_point.resize_(_zero_point.shape)
                self.scale.copy_(_scale)
                self.zero_point.copy_(_zero_point)
                self.scale = self.scale.to(self.device).to(_scale.dtype)
                self.zero_point = self.zero_point.to(self.device).to(_zero_point.dtype)

                # TODO: Remove amax for fp8 quant & export
                _amax = self.observer.amax()
                if self.amax.shape != _amax.shape:
                    self.amax.resize_(_amax.shape)
                self.amax.copy_(_amax)

            # Do fake quantize
            if self.fake_quant_enabled[0] == 1:
                X = quark.torch.kernel.quant_dequant_fp8_e4m3(X, self.scale)  # type: ignore[attr-defined]

        return X

    def extra_repr(self) -> str:
        if self.dtype == Dtype.fp8_e4m3:
            return 'fake_quant_enabled={}, observer_enabled={}, ' \
                   'maxbound={}, dtype={}, qscheme={}, ch_axis={}, '.format(
                       self.fake_quant_enabled, self.observer_enabled,
                       self.maxbound, self.dtype, self.qscheme, self.ch_axis)

        else:
            return 'fake_quant_enabled={}, observer_enabled={}, ' \
                   'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
                   'scale={}, zero_point={}'.format(
                       self.fake_quant_enabled, self.observer_enabled,
                       self.quant_min, self.quant_max,
                       self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination: Dict[str, Union[torch.nn.Parameter, torch.Tensor]], prefix: str,
                            keep_vars: bool) -> None:
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)  # type: ignore
        if self.dtype == Dtype.fp8_e4m3:
            destination[prefix + 'amax'] = self.amax
        elif self.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:
            destination[prefix + 'scale'] = self.scale
            destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict: Dict[str, Union[torch.nn.Parameter, torch.Tensor]], prefix: str,
                              local_metadata: Dict[str, Any], strict: bool, missing_keys: List[str],
                              unexpected_keys: List[str], error_msgs: List[str]) -> None:
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():  # type: ignore[attr-defined]
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore

    def export_amax(self) -> Optional[torch.Tensor]:
        """
        Adapter for GPU export
        """
        # Support fp8 so far
        assert self.dtype == Dtype.fp8_e4m3
        if self.amax is None:
            return None
        else:
            if not hasattr(self, "_amax_shape_for_export"):
                amax: torch.Tensor = self.amax
            else:
                amax = self.amax.reshape(self._amax_shape_for_export)
            amax[amax == 0] = self.maxbound
            return amax

    @property
    def num_bits(self) -> Union[int, Tuple[int, int]]:
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value: int) -> None:
        self._num_bits = value

    @property
    def block_sizes(self) -> int:
        """Return block_sizes for quantization."""
        assert self.group_size, "Invalid block_sizes"
        return self.group_size

    @block_sizes.setter
    def block_sizes(self, value: int) -> None:
        self._axis = None
        self.group_size = value


class FreezedFakeQuantize(nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, ) -> None:
        super(FreezedFakeQuantize, self).__init__()
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.quant_min: Optional[int] = None
        self.quant_max: Optional[int] = None
        self.dtype: Optional[Dtype] = None
        self.qscheme: Optional[QSchemeType] = None
        self.ch_axis: Optional[int] = None
        self.group_size: Optional[int] = None
        self.round_method: Optional[RoundType] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.dtype in [Dtype.bfloat16, Dtype.float16]:
            quant_dtype = {Dtype.bfloat16: torch.bfloat16, Dtype.float16: torch.float16}.get(self.dtype)
            input_origin_type = X.dtype
            if input_origin_type != quant_dtype:
                X = X.to(quant_dtype)
                X = X.to(input_origin_type)
        elif self.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:
            # Reshape for per_group input. TODO: Remove this and write a new kernel named fake_quantize_per_group_affine
            if self.qscheme in [QSchemeType.per_group]:
                org_x_shape = X.shape
                if self.group_size and self.group_size > 0:
                    assert org_x_shape[-1] % self.group_size == 0
                    X = X.reshape(-1, self.group_size)
                assert X.dim() == 2
                self.scale = self.scale.reshape(-1)
                self.zero_point = self.zero_point.reshape(-1)

            if self.qscheme == QSchemeType.per_tensor:
                X = quark.torch.kernel.fake_quantize_per_tensor_affine(  # type: ignore[attr-defined]
                    X, self.scale, self.zero_point.to(torch.int), self.quant_min, self.quant_max, RoundMode.NEARBYINT)
            elif self.qscheme in [QSchemeType.per_channel, QSchemeType.per_group]:
                assert self.ch_axis is not None
                X = quark.torch.kernel.fake_quantize_per_channel_affine(  # type: ignore[attr-defined]
                    X, self.scale, self.zero_point.to(torch.int), self.ch_axis, self.quant_min, self.quant_max,
                    RoundMode.NEARBYINT)
            else:
                ValueError(f"Do not support QSchema as {self.qscheme}")

            # Reshape back for per_group input. TODO: Write a new kernel named fake_quantize_per_group_affine
            if self.qscheme in [QSchemeType.per_group]:
                X = X.reshape(org_x_shape)
                self.scale = self.scale.reshape(X.shape[0], -1)
                self.zero_point = self.zero_point.reshape(X.shape[0], -1)
        elif self.dtype == Dtype.fp8_e4m3:
            X = quark.torch.kernel.quant_dequant_fp8_e4m3(X, self.scale)  # type: ignore[attr-defined]
        return X

    @classmethod
    def from_fake_quantize(cls, fake_quantize_model: FakeQuantize) -> nn.Module:
        freezed_fake_quantize_model = cls()
        freezed_fake_quantize_model.dtype = fake_quantize_model.dtype
        if fake_quantize_model.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.fp8_e4m3]:
            freezed_fake_quantize_model.register_buffer('scale', fake_quantize_model.scale)
            freezed_fake_quantize_model.register_buffer('zero_point', fake_quantize_model.zero_point)
        freezed_fake_quantize_model.qscheme = fake_quantize_model.qscheme
        freezed_fake_quantize_model.ch_axis = fake_quantize_model.ch_axis
        freezed_fake_quantize_model.group_size = fake_quantize_model.group_size
        freezed_fake_quantize_model.round_method = fake_quantize_model.round_method
        freezed_fake_quantize_model.quant_min = getattr(fake_quantize_model, 'quant_min', None)
        freezed_fake_quantize_model.quant_max = getattr(fake_quantize_model, 'quant_max', None)
        return freezed_fake_quantize_model


class SequentialFakeQuantize(nn.Sequential):
    pass
