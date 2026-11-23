"""Tests for ``mon.nn.complexity``."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mon.nn.complexity import count_flops, measure_latency, measure_peak_memory, profile_complexity


def _conv_macs(c_in: int, c_out: int, k: int, h: int, w: int, groups: int = 1) -> int:
    return c_out * (c_in // groups) * k * k * h * w


def test_count_flops_conv_and_linear():
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1, groups=4),
                          nn.BatchNorm2d(16), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 10)).eval()
    macs = _conv_macs(3, 16, 3, 8, 8) + _conv_macs(16, 16, 3, 8, 8, groups=4) + 16 * 10
    assert count_flops(model, torch.zeros(2, 3, 8, 8)) == 2 * 2 * macs  # batch 2; BN/ReLU/pool count zero


def test_count_flops_sees_functional_conv():
    weight = torch.randn(4, 3, 3, 3)
    flops = count_flops(lambda x: F.conv2d(x, weight, padding=1), torch.zeros(1, 3, 5, 5))
    assert flops == 2 * _conv_macs(3, 4, 3, 5, 5)


def test_zero_dce_macs_per_pixel():
    from mon.models.registry import load

    model = load("zero_dce", weights=None)
    c = profile_complexity(model, torch.zeros(1, 3, 32, 48), latency=False)
    # 7 conv3x3 layers: 3->32, 3x 32->32, 2x 64->32, 64->24 (curve maps); elementwise curve steps count zero
    per_pixel = 9 * (3 * 32 + 3 * 32 * 32 + 2 * 64 * 32 + 64 * 24)
    assert (c.params, c.macs, c.flops) == (79_416, per_pixel * 32 * 48, 2 * per_pixel * 32 * 48)
    assert c.input_shapes == [[1, 3, 32, 48]] and c.device == "cpu"


def test_profile_latency_and_memory_cpu():
    c = profile_complexity(nn.Conv2d(3, 4, 3), torch.zeros(1, 3, 16, 16), warmup=1, runs=3)
    assert c.latency_ms is not None and c.latency_ms > 0 and c.latency_median_ms is not None
    assert c.latency_std_ms is not None and c.peak_memory is None  # no peak counter off CUDA
    assert set(c.to_dict()) >= {"params", "flops", "macs", "latency_ms", "peak_memory"}


def test_profile_parameter_free_and_multiple_inputs():
    class Add(nn.Module):
        def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

    c = profile_complexity(Add(), [torch.zeros(2, 2), torch.ones(2, 2)], latency=False)
    assert (c.params, c.flops) == (0, 0) and c.input_shapes == [[2, 2], [2, 2]]


def test_measure_latency_counts_and_validates():
    calls: list[int] = []
    times = measure_latency(lambda x: calls.append(1), torch.zeros(1), warmup=2, runs=4)
    assert len(times) == 4 and len(calls) == 6
    with pytest.raises(ValueError, match="runs >= 1"):
        measure_latency(lambda x: None, torch.zeros(1), runs=0)


def test_profile_rejects_device_mismatch():
    with pytest.raises(ValueError, match="model is on"):
        profile_complexity(nn.Linear(2, 2), torch.zeros(1, 2, device="meta"), latency=False)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_peak_memory_cuda():
    device = torch.device("cuda")
    x = torch.zeros(1, 3, 64, 64, device=device)
    peak = measure_peak_memory(nn.Conv2d(3, 8, 3).to(device), x, device)
    assert peak is not None and peak >= x.numel() * 4
