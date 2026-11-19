"""Measure a registered model's complexity: parameters, FLOPs/MACs, latency, peak memory.

``benchmark(cfg)`` builds ``cfg.model`` (random init by default: complexity doesn't depend on the
weights, and nothing needs downloading), runs it on a seeded random ``[batch, 3, H, W]`` input
(``[batch, stack, 3, H, W]`` for exposure-stack models), and returns
``mon.nn.complexity.Complexity`` as a dict with the resolved config under ``config``. ``H`` and
``W`` are rounded up to the model's ``input_multiple``. The result is printed; with ``output``,
it is also written there as YAML. For a zero-shot model this measures one forward pass of the
network, not the per-image fitting.

Examples:
    >>> benchmark(BenchmarkConfig(model="zero_dce", size=[512, 512]))  # doctest: +SKIP
    {'params': 79416, 'flops': 41523609600, 'macs': 20761804800, ...}
"""

__all__ = [
    "BenchmarkConfig",
    "benchmark",
]

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from omegaconf import MISSING, DictConfig, OmegaConf

from mon.core.console import get_logger, print_dict
from mon.core.enums import Task
from mon.core.fileio import save_yaml
from mon.core.path import Path
from mon.core.runtime import seed_everything, select_device
from mon.models.registry import MODELS, load
from mon.nn.complexity import profile_complexity

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Settings for ``benchmark`` (also the schema of ``mon benchmark`` configs).

    Attributes:
        model: ``MODELS`` name.
        weights: ``load`` spec; ``None`` (default) is the constructor's random init.
        model_args: Constructor arguments, e.g. ``{"scale_factor": 12}`` for Zero-DCE++.
        size: Input ``[H, W]`` (one value: square), rounded up to the model's ``input_multiple``.
        batch_size: Input batch size.
        stack: Images per exposure stack, for exposure-stack (MEF) models only.
        latency: Time the forward pass.
        memory: Measure peak CUDA memory (``None`` on other devices).
        warmup: Untimed forward passes before timing.
        runs: Timed forward passes.
        device: ``select_device`` spec.
        seed: Seed for the random input (``None``: drawn and recorded).
        output: YAML file for the result; ``None``: only printed.
        exist_ok: Overwrite ``output`` if it exists (otherwise that raises).
    """

    model     : str = MISSING
    weights   : str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    size      : list[int] = field(default_factory=lambda: [512, 512])
    batch_size: int = 1
    stack     : int = 3
    latency   : bool = True
    memory    : bool = True
    warmup    : int = 5
    runs      : int = 20
    device    : str = "auto"
    seed      : int | None = None
    output    : str | None = None
    exist_ok  : bool = False


# --- Helpers ---

def _as_config(cfg: BenchmarkConfig | DictConfig | Mapping[str, Any]) -> BenchmarkConfig:
    merged = OmegaConf.merge(OmegaConf.structured(BenchmarkConfig), cfg)
    if OmegaConf.is_missing(merged, "model"):
        raise ValueError("BenchmarkConfig.model is required")
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, BenchmarkConfig)
    if len(obj.size) not in (1, 2) or any(s < 1 for s in obj.size):
        raise ValueError(f"size must be [H, W] or [S] with positive values, got {obj.size}")
    if obj.batch_size < 1 or obj.stack < 1:
        raise ValueError(f"batch_size and stack must be >= 1, got {obj.batch_size}, {obj.stack}")
    if obj.runs < 1 or obj.warmup < 0:
        raise ValueError(f"need runs >= 1 and warmup >= 0, got runs={obj.runs}, warmup={obj.warmup}")
    return obj


def _input_size(size: list[int], multiple: int) -> tuple[int, int]:
    h, w = (size[0], size[0]) if len(size) == 1 else (size[0], size[1])
    return math.ceil(h / multiple) * multiple, math.ceil(w / multiple) * multiple


# --- Public API ---

def benchmark(cfg: BenchmarkConfig | DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Profile ``cfg.model`` on a random input; return ``Complexity.to_dict()`` plus ``config``.

    Raises:
        ValueError: For a missing ``model`` or invalid sizes/counts.
        RegistryKeyError: If ``model`` isn't registered.
        FileExistsError: If ``output`` exists and ``exist_ok`` is false.
    """
    config = _as_config(cfg)
    entry  = MODELS.entry(config.model)
    output = Path(config.output).expanduser() if config.output is not None else None
    if output is not None and output.exists() and not config.exist_ok:
        raise FileExistsError(f"output {output} exists; pass exist_ok=true to overwrite it")

    seed     = seed_everything(config.seed)
    device   = select_device(config.device)
    multiple = int(getattr(entry.obj, "input_multiple", 1))
    h, w     = _input_size(config.size, multiple)
    if (h, w) != _input_size(config.size, 1):
        logger.info("size %s rounded up to %dx%d (%s input_multiple=%d)", config.size, h, w, config.model, multiple)
    if entry.meta.get("zero_shot", False):
        logger.warning("%s is zero-shot: this measures one network forward pass, not the per-image fitting",
                       config.model)

    model = load(config.model, config.weights, device=device, **config.model_args)
    shape = ((config.batch_size, config.stack, 3, h, w) if Task.MEF in entry.meta.get("tasks", set())
             else (config.batch_size, 3, h, w))
    x = torch.rand(shape, device=device)

    result = profile_complexity(model, x, latency=config.latency, memory=config.memory,
                                warmup=config.warmup, runs=config.runs).to_dict()
    result["config"] = {**asdict(config), "seed": seed, "device": str(device)}
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)

    print_dict(_summary(result), title=f"Complexity: {config.model}")
    if output is not None:
        save_yaml(result, output)
        logger.info("wrote %s", output)
    return result


def _summary(result: Mapping[str, Any]) -> dict[str, str]:
    """Human-readable view of a ``benchmark`` result."""
    def ms(v: float | None) -> str:
        return "-" if v is None else f"{v:.3f} ms"

    peak = result["peak_memory"]
    return {
        "input"  : " × ".join(map(str, result["input_shapes"][0])),
        "device" : str(result.get("device_name", result["device"])),
        "params" : f"{result['params']:,} ({result['params'] / 1e6:.4f} M)",
        "MACs"   : f"{result['macs'] / 1e9:.4f} G",
        "FLOPs"  : f"{result['flops'] / 1e9:.4f} G",
        "latency": f"{ms(result['latency_ms'])} (median {ms(result['latency_median_ms'])})",
        "peak"   : "-" if peak is None else f"{peak / 2**20:.1f} MiB",
    }
