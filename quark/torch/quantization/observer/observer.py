#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
if TYPE_CHECKING:
    from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.nn.utils import check_min_max_valid
from quark.torch.quantization.utils import calculate_qmin_qmax


class ObserverBase(ABC, nn.Module):

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__()
        self.dtype = qspec.dtype

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        pass


class PlaceholderObserver(ObserverBase):
    r"""
    Observer only passes its configuration to the quantized module's ``.from_float()``.

    Does not have any calculation.

    Only can be used for quantization to float16 and bfloat16 which doesn't require determining
    ranges.
    """

    def __init__(
        self,
        qspec: QuantizationSpec,
    ) -> None:
        super().__init__(qspec)

        assert self.dtype in [Dtype.bfloat16, Dtype.float16
                              ], "PlaceholderObserver can only be used for bfloat16 and float16 quantization"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def extra_repr(self) -> str:
        return f"dtype={self.dtype}"


class UniformScalingObserver(ObserverBase):
    """
    Observer for uniform scaling quantizer. For example 'int uniform quantizer' or 'fp8 uniform scaling'.

    """

    eps: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, qspec: QuantizationSpec, eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec)

        self.qspec = qspec
        self.symmetric = qspec.symmetric
        self.scale_type = qspec.scale_type
        self.qscheme = qspec.qscheme

        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("eps", torch.tensor([eps]))

        self.quant_min, self.quant_max = calculate_qmin_qmax(qspec.dtype)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        return self.calculate_qparams(self.min_val, self.max_val)

    def calculate_qparams(self, min_val: torch.Tensor, max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        if self.dtype in [Dtype.fp8_e4m3]:
            return self.calculate_fp8_quant_parameters(min_val, max_val)
        else:
            return self.calculate_int_quant_params(min_val, max_val)

    def calculate_int_quant_params(self, min_val: torch.Tensor,
                                   max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # TODO setup eps device when init
        self.eps = self.eps.to(min_val.dtype).to(min_val.device)

        if not check_min_max_valid(min_val, max_val):
            return torch.tensor([1.0], device=min_val.device.type), torch.tensor([0], device=min_val.device.type)

        quant_min, quant_max = self.quant_min, self.quant_max
        assert isinstance(quant_min, int)
        assert isinstance(quant_max, int)
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        if self.symmetric:
            max_val_pos = torch.max(-min_val_neg, max_val_pos)
            scale = max_val_pos / (float(quant_max - quant_min) / 2)
            scale = torch.max(scale, self.eps)
        else:
            # AWQ
            scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
            # TODO: reset eps's device
            self.eps = self.eps.to(scale.device)
            scale = torch.max(scale, self.eps)
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)

        return scale, zero_point

    def calculate_fp8_quant_parameters(self, min_val: torch.Tensor,
                                       max_val: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int32, device=device)

        amax = torch.maximum(torch.abs(min_val_neg), torch.abs(max_val_pos))
        scale = amax / 448
        return scale, zero_point

    def extra_repr(self) -> str:
        return f"min_val={self.min_val}, max_val={self.max_val}"

    def reset_min_max_vals(self) -> None:
        """Resets the min/max values."""
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))


class PerTensorMinMaxObserver(UniformScalingObserver):

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__(qspec)

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig

        x = x_orig.detach()  # avoid keeping autograd tape
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.input_origin_type = x_orig.dtype
        self.min_val = self.min_val.to(self.input_origin_type)
        self.max_val = self.max_val.to(self.input_origin_type)
        return x_orig

    # TODO: Delete amax function
    def amax(self) -> torch.Tensor:
        return torch.maximum(torch.abs(self.min_val), torch.abs(self.max_val))


class PerChannelMinMaxObserver(UniformScalingObserver):

    def __init__(self, qspec: QuantizationSpec, eps: float = torch.finfo(torch.float32).eps) -> None:
        super().__init__(qspec)

        self.qspec = qspec
        self.ch_axis = qspec.ch_axis
        self.group_size = qspec.group_size

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        return self._forward(x_orig)

    def _forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        self.device = x_orig.device
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_val = self.min_val.to(self.device)
        max_val = self.max_val.to(self.device)
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        if self.ch_axis is not None:
            new_axis_list[self.ch_axis] = 0
        else:
            raise ValueError("ch_axis cannot be None")
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        # Need to match dtype of min/max because the updates to buffers
        # are done in place and types need to match for comparisons
        y = y.to(self.min_val.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_val.numel() == 0 or max_val.numel() == 0:
            min_val, max_val = torch.aminmax(y, dim=1)
        else:
            min_val_cur, max_val_cur = torch.aminmax(y, dim=1)
            min_val = torch.min(min_val_cur, min_val)
            max_val = torch.max(max_val_cur, max_val)
        self.min_val.resize_(min_val.shape)
        self.max_val.resize_(max_val.shape)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.input_origin_type = x_orig.dtype
        self.min_val = self.min_val.to(self.input_origin_type)
        self.max_val = self.max_val.to(self.input_origin_type)
        return x_orig


class PerTensorHistogramObserver(UniformScalingObserver):

    calib_bin_edges: torch.Tensor
    calib_hist: torch.Tensor

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__(qspec)

        self.register_buffer("calib_bin_edges", torch.tensor([]))
        self.register_buffer("calib_hist", torch.tensor([]))

        # TODO: make the value can be set
        self._skip_zeros = False
        self._num_bins = 2048

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        """
        Records the running histogram of ``x_orig``.

        Raises:
        - ValueError: If the `self.symmetric` argument is False.

        """
        self.device = x_orig.device
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.float()

        with torch.no_grad():
            if self._skip_zeros:
                x = x[torch.where(x != 0)]

            assert isinstance(x, torch.Tensor)
            if self.symmetric is not None and self.symmetric is False:
                x_max = x.max().item()
                x_min = x.min().item()
            else:
                if torch.min(x) < 0.0:
                    x = x.abs()
                x_max = x.max().item()
                x_min = 0.0

            if self.calib_bin_edges.nelement() == 0 and self.calib_hist.nelement() == 0:
                self.calib_hist = torch.histc(x, bins=self._num_bins, min=x_min, max=x_max)
                self.calib_bin_edges = torch.linspace(x_min, x_max, self._num_bins + 1)
            else:
                if x_max > self.calib_bin_edges[-1]:
                    width = (self.calib_bin_edges[1] - self.calib_bin_edges[0]).item()
                    self._num_bins += int(((x_max - self.calib_bin_edges[-1]) / width).ceil().item())
                    self.calib_bin_edges = torch.arange(self.calib_bin_edges[0].item(),
                                                        x_max + width,
                                                        width,
                                                        device=x.device)

                if x_min < self.calib_bin_edges[0]:
                    width = (self.calib_bin_edges[1] - self.calib_bin_edges[0]).item()
                    self._num_bins += int(((self.calib_bin_edges[0] - x_min) / width).ceil().item())
                    self.calib_bin_edges = torch.arange(x_min - width,
                                                        self.calib_bin_edges[-1].item(),
                                                        width,
                                                        device=x.device)

                assert x_max <= self.calib_bin_edges[-1]
                assert x_min >= self.calib_bin_edges[0]

                hist = torch.histc(x,
                                   bins=self._num_bins,
                                   min=self.calib_bin_edges[0].item(),
                                   max=self.calib_bin_edges[-1].item())
                hist[:self.calib_hist.numel()] += self.calib_hist
                self.calib_hist = hist

            assert isinstance(self.calib_hist, torch.Tensor)
            assert isinstance(self.calib_bin_edges, torch.Tensor)
            self.calib_hist = self.calib_hist.to(self.device)
            self.calib_bin_edges = self.calib_bin_edges.to(self.device)

        return x_orig


class PerTensorPercentileObserver(PerTensorHistogramObserver):

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__(qspec)

        # TODO: make the value can be set
        self._skip_zeros = True
        self._num_bins = 4200
        self.percentile = 99.99999999

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        self.min_val, self.max_val = self._calculate_min_and_max_using_percentile()
        return self.calculate_qparams(self.min_val, self.max_val)

    def _calculate_min_and_max_using_percentile(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.calib_hist, torch.Tensor)
        assert isinstance(self.calib_bin_edges, torch.Tensor)
        return self.get_min_max_by_percentile(self.calib_hist, self.calib_bin_edges, self.percentile)

    def get_min_max_by_percentile(self, histogram: torch.Tensor, bin_edges: torch.Tensor,
                                  percentile: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the minimum and maximum values of a histogram at a specified percentile.

        Parameters:
        - histogram (torch.Tensor): A tensor representing the histogram of the data. Each element
        in the histogram represents the frequency of data in the corresponding bin.
        - bin_edges (torch.Tensor): A tensor containing the edge values that correspond to the
        bins represented in the histogram. There should be one more element in `bin_edges` than
        in `histogram`.
        - percentile (int): The percentile at which to determine the minimum and maximum values.
        The value should be an integer between 0 and 100.

        Returns:
        - Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors. The first tensor
        is the value at the specified percentile, and the second tensor is the value at the
        complementary percentile (i.e., 100-percentile).

        Raises:
        - ValueError: If the `percentile` argument is not within the range 0 to 100.
        """
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile value must be between 0 and 100.")

        # Return None if no data is available
        if bin_edges is None and histogram is None:
            return None

        # Calculate cumulative distribution function
        hist_total = histogram.sum()
        cumulative_dist = torch.cumsum(histogram / hist_total, dim=0)

        if self.symmetric is not None and self.symmetric is False:
            target_pct_one_side = (100.0 - percentile) / 200.0

            upper_idx = (cumulative_dist >= target_pct_one_side).nonzero().min().item()
            assert isinstance(upper_idx, int), "Index must be an integer"
            max_value = bin_edges[upper_idx]

            lower_idx = (cumulative_dist <= (1 - target_pct_one_side)).nonzero().min().item()
            assert isinstance(lower_idx, int), "Index must be an integer"
            min_value = bin_edges[lower_idx]

        else:
            target_pct = percentile / 100.0
            cumulative_dist_max = cumulative_dist[-1].item()
            assert isinstance(cumulative_dist_max, float)
            target_pct = min(target_pct, cumulative_dist_max)

            upper_idx = (cumulative_dist >= target_pct).nonzero().min().item()
            assert isinstance(upper_idx, int), "Index must be an integer"
            max_value = bin_edges[upper_idx]

            min_value = torch.tensor(0, device='cpu')

        max_value = max_value.to(self.device)
        min_value = min_value.to(self.device)
        return min_value, max_value

    # TODO: Delete amax function
    def amax(self) -> torch.Tensor:
        return self._calculate_min_and_max_using_percentile()[0]


class PerTensorMSEObserver(PerTensorHistogramObserver):

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__(qspec)

    def _calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.min_val, self.max_val = self._calculate_min_and_max_using_mse()
        return self.calculate_qparams(self.min_val, self.max_val)

    def _calculate_min_and_max_using_mse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.calib_hist, torch.Tensor)
        assert isinstance(self.calib_bin_edges, torch.Tensor)
        return self.get_min_max_by_mse(self.calib_hist, self.calib_bin_edges)

    def get_min_max_by_mse(self,
                           calib_hist: torch.Tensor,
                           calib_bin_edges: torch.Tensor,
                           stride: int = 1,
                           start_bin: int = 2045) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns amax that minimizes MSE of the collected histogram."""
        # If calibrator hasn't collected any data, return none
        if calib_bin_edges is None and calib_hist is None:
            return None

        counts = calib_hist
        edges = calib_bin_edges

        counts = counts.to(self.device)
        edges = edges.to(self.device)

        centers = (edges[1:] + edges[:-1]) / 2

        mses = []
        arguments = []

        min_value = torch.tensor(0, device='cpu')
        min_value = min_value.to(self.device)

        for i in range(start_bin, len(centers), stride):
            amax = centers[i]
            if self.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8]:
                quant_centers = self.int_fake_tensor_quant(centers, min_value, amax)
            elif self.dtype in [Dtype.fp8_e4m3]:
                quant_centers = self.scaling_fp8e4m3(centers, amax)
            else:
                raise TypeError("Invalid dtype. dtype must be a positive integer or fp8_e4m3.")

            mse = ((quant_centers - centers)**2 * counts).mean()

            mses.append(mse.cpu())
            arguments.append(i)

        argmin = torch.argmin(torch.stack(mses))
        calib_amax = centers[arguments[argmin]]

        calib_amax = calib_amax.to(self.device)

        return min_value, calib_amax

    def int_fake_tensor_quant(self, X: torch.Tensor, min_value: torch.Tensor, max_value: torch.Tensor) -> torch.Tensor:
        scale, zero_point = self.calculate_int_quant_params(min_value, max_value)
        assert isinstance(self.quant_min, int)
        assert isinstance(self.quant_max, int)
        X = torch.fake_quantize_per_tensor_affine(X, scale.to(torch.float), zero_point.to(torch.int), self.quant_min,
                                                  self.quant_max)
        return X

    def scaling_fp8e4m3(self, X: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
        X_orig_dtype = X.dtype
        scale = amax / 448.0
        X = X / scale
        X = torch.clamp(X, min=-448, max=448)
        X = X.to(torch.float8_e4m3fn).to(X_orig_dtype) * scale
        return X

    # TODO: Delete amax function
    def amax(self) -> torch.Tensor:
        return self._calculate_min_and_max_using_mse()[0]
