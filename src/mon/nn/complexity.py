"""Model complexity: parameters, FLOPs/MACs, latency, and peak memory.

FLOPs come from ``torch.utils.flop_counter.FlopCounterMode``, which counts matrix
multiplications and convolutions at the ATen level (so functional ``F.conv2d`` calls count
too) and reports ``2 × MACs`` for them; elementwise ops, normalization, pooling, activations,
and interpolation count zero. thop (used by ``mon_old``) hooks modules instead: it misses
functional convolutions but adds small terms for BatchNorm, pooling, and upsampling, so its
totals can differ slightly. Many low-light papers label thop's MACs "FLOPs": compare ``macs``
with those. Adaptive solvers (CLODE's ODE) make the count input-dependent.

References:
    - ``mon_old/libs/mon/src/mon/metrics/complexity.py`` (thop-based predecessor)
"""

__all__ = [
    "Complexity",
    "count_flops",
    "measure_latency",
    "measure_peak_memory",
    "profile_complexity",
]

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.flop_counter import FlopCounterMode

from mon.core.timing import synchronize
from mon.nn.utils import count_parameters, module_device

type Inputs = Tensor | Sequence[Tensor]


def _as_args(inputs: Inputs) -> tuple[Tensor, ...]:
    return (inputs,) if isinstance(inputs, Tensor) else tuple(inputs)


def _devices(inputs: tuple[Tensor, ...]) -> set[torch.device]:
    return {t.device for t in inputs}


@dataclass(frozen=True)
class Complexity:
    """Result of ``profile_complexity``.

    Attributes:
        params: Number of parameters (shared ones counted once).
        flops: Floating-point operations of one forward pass (``2 × macs``).
        macs: Multiply-accumulates of one forward pass.
        latency_ms: Mean wall-clock time of one forward pass (ms); ``None`` if not measured.
        latency_median_ms: Median of the timed runs (ms); ``None`` if not measured.
        latency_std_ms: Sample standard deviation of the timed runs (ms); ``None`` if not
            measured or fewer than two runs.
        peak_memory: Peak CUDA memory allocated during a forward pass (bytes), including the
            weights and inputs already on the device; ``None`` off CUDA or if not measured.
        input_shapes: Shape of each input tensor.
        device: Device the model ran on.
    """

    params           : int
    flops            : int
    macs             : int
    latency_ms       : float | None
    latency_median_ms: float | None
    latency_std_ms   : float | None
    peak_memory      : int | None
    input_shapes     : list[list[int]]
    device           : str

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (YAML/JSON friendly)."""
        return asdict(self)


def count_flops(fn: Callable[..., Any], inputs: Inputs) -> int:
    """FLOPs of ``fn(*inputs)`` under ``inference_mode`` (matmuls and convolutions, ``2 × MACs``).

    Examples:
        >>> conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        >>> count_flops(conv, torch.zeros(1, 3, 4, 4)) == 2 * (8 * 3 * 3 * 3) * (4 * 4)
        True
    """
    args    = _as_args(inputs)
    counter = FlopCounterMode(display=False)
    with torch.inference_mode(), counter:
        fn(*args)
    return int(counter.get_total_flops())


def measure_latency(fn: Callable[..., Any], inputs: Inputs, *, warmup: int = 5, runs: int = 20) -> list[float]:
    """Wall-clock time (ms) of each of ``runs`` calls of ``fn(*inputs)`` after ``warmup`` untimed calls.

    Each call runs under ``inference_mode`` and is bracketed by ``synchronize()``, so queued
    CUDA/MPS work is counted in the call that queued it.

    Raises:
        ValueError: If ``runs < 1`` or ``warmup < 0``.
    """
    if runs < 1 or warmup < 0:
        raise ValueError(f"need runs >= 1 and warmup >= 0, got runs={runs}, warmup={warmup}")
    args = _as_args(inputs)
    times: list[float] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args)
        for _ in range(runs):
            synchronize()
            start = time.perf_counter()
            fn(*args)
            synchronize()
            times.append((time.perf_counter() - start) * 1e3)
    return times


def measure_peak_memory(fn: Callable[..., Any], inputs: Inputs, device: torch.device) -> int | None:
    """Peak CUDA memory (bytes) allocated on ``device`` while running ``fn(*inputs)`` once.

    The peak includes everything already allocated there (weights, inputs). Returns ``None``
    for non-CUDA devices: CPU and MPS expose no peak-allocation counter.
    """
    if device.type != "cuda":
        return None
    args = _as_args(inputs)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        fn(*args)
    torch.cuda.synchronize(device)
    return int(torch.cuda.max_memory_allocated(device))


def profile_complexity(
    model  : nn.Module,
    inputs : Inputs,
    *,
    latency: bool = True,
    memory : bool = True,
    warmup : int  = 5,
    runs   : int  = 20,
) -> Complexity:
    """Parameters, FLOPs/MACs, and optionally latency and peak memory of ``model(*inputs)``.

    The model must be on one device, the same as the inputs. Call ``model.eval()`` first if it
    has dropout or BatchNorm. For a zero-shot model this measures one forward pass of the
    network, not the per-image fitting.

    Args:
        model: Module to profile.
        inputs: One tensor or a sequence of tensors, passed positionally.
        latency: Time ``runs`` forward passes after ``warmup`` (see ``measure_latency``).
        memory: Measure peak CUDA memory (see ``measure_peak_memory``).
        warmup: Untimed passes before timing.
        runs: Timed passes.

    Raises:
        ValueError: If the inputs aren't on the model's device (a parameter-free model takes
            the inputs' device), or the inputs span several devices.

    Examples:
        >>> c = profile_complexity(nn.Conv2d(3, 8, 3, padding=1), torch.zeros(1, 3, 8, 8), latency=False)
        >>> c.params, c.macs, c.flops, c.latency_ms
        (224, 13824, 27648, None)
    """
    args    = _as_args(inputs)
    devices = _devices(args)
    if len(devices) != 1:
        raise ValueError(f"inputs must be on one device, got {sorted(map(str, devices))}")
    device = devices.pop()
    if any(True for _ in (*model.parameters(), *model.buffers())):
        model_dev = module_device(model)
        if model_dev != device:
            raise ValueError(f"model is on {model_dev} but inputs are on {device}")

    flops = count_flops(model, args)
    times = measure_latency(model, args, warmup=warmup, runs=runs) if latency else None
    peak  = measure_peak_memory(model, args, device) if memory else None
    return Complexity(
        params            = count_parameters(model),
        flops             = flops,
        macs              = flops // 2,
        latency_ms        = statistics.fmean(times) if times else None,
        latency_median_ms = statistics.median(times) if times else None,
        latency_std_ms    = statistics.stdev(times) if times and len(times) > 1 else None,
        peak_memory       = peak,
        input_shapes      = [list(t.shape) for t in args],
        device            = str(device),
    )
