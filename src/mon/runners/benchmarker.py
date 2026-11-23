"""Measure a registered model's complexity: parameters, FLOPs/MACs, latency, peak memory.

``benchmark(cfg)`` builds ``cfg.model`` (random init by default: complexity doesn't depend on the
weights, and nothing needs downloading), runs it on a seeded random ``[batch, 3, H, W]`` input
(``[batch, stack, 3, H, W]`` for exposure-stack models), and returns
``mon.nn.complexity.Complexity`` as a dict with the resolved config under ``config``. ``H`` and
``W`` are rounded up to the model's ``input_multiple``. The result is printed; with ``output``,
it is also written there as YAML.

A zero-shot model (registry ``zero_shot=True``, e.g. ``colie``, ``zero_ig``) is measured as it runs at test time:
each timed run is one call of its registry ``fit`` function (a fresh network fitted to the input, ``fit_args``
passed through), so latency is the per-image fit time and peak memory is taken during the timed runs. MACs are
the whole fit's (forward and backward, every step), counted once per input size; that pass is also the warm-up,
so ``warmup`` defaults to 0 and ``runs`` to 1. ``zero_shot=false`` measures the network's forward pass instead
(e.g. Zero-IG's released weights).

With ``data`` (a ``DATASETS`` id, e.g. ``testkit_efficiency``, or a folder of images), it measures real
images instead, grouped by resolution: per size, MACs (mean over the images: adaptive models such as
CLODE vary with content), latency over every image (``warmup`` untimed passes on the first image of
the size, then ``runs`` timed passes per image), and peak CUDA memory (max over the images). A size
that runs out of memory (or fails) is recorded with its error and the next size runs. The result
(``{"model", "params", "rows", "config"}``) is printed as a table; with ``output``, written as YAML plus
a CSV of the rows next to it.

Examples:
    >>> benchmark(BenchmarkConfig(model="zero_dce", size=[512, 512]))  # doctest: +SKIP
    {'params': 79416, 'flops': 41523609600, 'macs': 20761804800, ...}
    >>> benchmark({"model": "zero_dce", "data": "testkit_efficiency", "output": "zero_dce.yaml"})  # doctest: +SKIP
    {'model': 'zero_dce', 'params': 79416, 'rows': [{'size': '256x256', 'macs': 5190451200.0, ...}, ...], ...}
    >>> benchmark({"model": "colie", "data": "testkit_efficiency"})  # per-image fit time  # doctest: +SKIP
"""

__all__ = [
    "BenchmarkConfig",
    "benchmark",
]

import csv
import importlib
import inspect
import io
import math
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from rich.table import Table
from torch import Tensor, nn
from torchvision.transforms.v2.functional import to_dtype

from mon.core.console import console, get_logger, print_dict
from mon.core.enums import Task
from mon.core.fileio import _atomic_write_text, save_yaml
from mon.core.path import Path
from mon.core.runtime import seed_everything, select_device
from mon.core.timing import synchronize
from mon.data.dataset import DATASETS, ImageFolder
from mon.data.transforms import pad_to_multiple
from mon.models.registry import MODELS, load
from mon.nn.complexity import count_flops, measure_latency, measure_peak_memory, profile_complexity
from mon.nn.utils import count_parameters

logger = get_logger(__name__)


@dataclass
class BenchmarkConfig:
    """Settings for ``benchmark`` (also the schema of ``mon benchmark`` configs).

    Attributes:
        model: ``MODELS`` name.
        weights: ``load`` spec; ``None`` (default) is the constructor's random init. Unused when fitting.
        model_args: Constructor arguments, e.g. ``{"scale_factor": 12}`` for Zero-DCE++.
        zero_shot: Time the registry ``fit`` function (per-image fitting) instead of the forward pass. ``None``:
            the registry's ``zero_shot`` flag (as ``mon predict``).
        fit_args: Keyword arguments for the ``fit`` function, e.g. ``{"iterations": 100}`` for Zero-IG. Those
            that are also constructor arguments (CoLIE's ``hidden_dim``) set the network whose parameters are counted.
        data: Measure real images instead of a random input: a ``DATASETS`` id (``split`` picks the split; e.g.
            ``testkit_efficiency``) or a folder (searched recursively). Grouped by resolution; ``size`` is unused.
        split: Dataset split when ``data`` is a dataset id.
        sizes: With ``data``, only these resolutions (``"<W>x<H>"``, e.g. ``["256x256", "1920x1080"]``).
        size: Input ``[H, W]`` (one value: square), rounded up to the model's ``input_multiple``.
        batch_size: Input batch size.
        stack: Images per exposure stack, for exposure-stack (MEF) models only.
        latency: Time the forward pass (or fit).
        memory: Measure peak CUDA memory (``None`` on other devices).
        warmup: Untimed passes before timing (with ``data``: per resolution). ``None``: 5, or 0 when fitting.
        runs: Timed passes (with ``data``: per image). ``None``: 20, or 1 when fitting.
        device: ``select_device`` spec.
        seed: Seed for the random input (``None``: drawn and recorded).
        output: YAML file for the result (with ``data``, also ``<stem>.csv``); ``None``: only printed.
        exist_ok: Overwrite ``output`` if it exists (otherwise that raises).
    """

    model     : str = MISSING
    weights   : str | None = None
    model_args: dict[str, Any] = field(default_factory=dict)
    zero_shot : bool | None = None
    fit_args  : dict[str, Any] = field(default_factory=dict)
    data      : str | None = None
    split     : str = "test"
    sizes     : list[str] | None = None
    size      : list[int] = field(default_factory=lambda: [512, 512])
    batch_size: int = 1
    stack     : int = 3
    latency   : bool = True
    memory    : bool = True
    warmup    : int | None = None
    runs      : int | None = None
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
    if (obj.runs is not None and obj.runs < 1) or (obj.warmup is not None and obj.warmup < 0):
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
    zero_shot = bool(entry.meta.get("zero_shot", False)) if config.zero_shot is None else config.zero_shot
    config.zero_shot = zero_shot
    config.warmup    = (0 if zero_shot else 5) if config.warmup is None else config.warmup
    config.runs      = (1 if zero_shot else 20) if config.runs is None else config.runs
    if zero_shot:
        if config.weights is not None and config.weights.strip().lower() != "default":
            logger.warning("weights=%r ignored: %s fits a fresh network per image (zero_shot=false runs weights)",
                           config.weights, config.model)
        if config.model_args:
            logger.warning("model_args only set the network whose parameters are counted; fit_args configure the fit")
    elif config.fit_args:
        logger.warning("fit_args ignored: %s runs its forward pass (zero_shot=false)", config.model)

    model_args = config.model_args
    if zero_shot:  # the fitted network's architecture follows fit_args (e.g. CoLIE's hidden_dim)
        accepted   = inspect.signature(entry.obj).parameters
        model_args = {k: v for k, v in config.fit_args.items() if k in accepted} | model_args
    model = load(config.model, None if zero_shot else config.weights, device=device, **model_args)
    fn    = _fit_fn(config, entry.meta.get("fit"), seed) if zero_shot else model
    if config.data is not None:
        if Task.MEF in entry.meta.get("tasks", set()):
            raise ValueError(f"{config.model!r} fuses exposure stacks; data= measures single images")
        return _benchmark_data(config, model, fn, multiple, device, seed, output)
    shape = ((config.batch_size, config.stack, 3, h, w) if Task.MEF in entry.meta.get("tasks", set())
             else (config.batch_size, 3, h, w))
    x = torch.rand(shape, device=device)

    if zero_shot:
        result = {"params": count_parameters(model), **_profile_fit(fn, x, config, device),
                  "input_shapes": [list(x.shape)], "device": str(device)}
    else:
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


# --- Zero-shot fitting ---

def _fit_fn(config: BenchmarkConfig, target: Any, seed: int) -> Callable[[Tensor], Any]:
    """``x -> fit(x, seed=seed, **fit_args)`` from the registry's ``fit`` target."""
    if not target:
        raise ValueError(f"{config.model!r} has no 'fit' function in MODELS; set zero_shot=false")
    module, _, name = str(target).partition(":")
    fit    = getattr(importlib.import_module(module), name)
    kwargs = {"seed": seed, **config.fit_args}
    return lambda x: fit(x, **kwargs)


def _run_fits(fn: Callable[[Tensor], Any], x: Tensor, config: BenchmarkConfig, device: torch.device,
              *, count: bool = True, warmup: int | None = None) -> tuple[int | None, list[float], list[int]]:
    """``(FLOPs or None, times in ms, peaks in bytes)``: one counted fit if ``count`` (also the warm-up), then
    ``warmup`` (default ``config.warmup``) untimed and ``config.runs`` timed fits (peak CUDA memory during each)."""
    assert config.warmup is not None and config.runs is not None
    warmup = config.warmup if warmup is None else warmup
    flops = count_flops(fn, x) if count else None
    cuda  = config.memory and device.type == "cuda"
    times: list[float] = []
    peaks: list[int] = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        for _ in range(config.runs if config.latency or cuda else 0):
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            synchronize()
            start = time.perf_counter()
            fn(x)
            synchronize()
            times.append((time.perf_counter() - start) * 1e3)
            if cuda:
                peaks.append(int(torch.cuda.max_memory_allocated(device)))
    return flops, times if config.latency else [], peaks


def _stats(times: list[float], peaks: list[int]) -> dict[str, Any]:
    return {
        "latency_ms"       : statistics.fmean(times) if times else None,
        "latency_median_ms": statistics.median(times) if times else None,
        "latency_std_ms"   : statistics.stdev(times) if len(times) > 1 else None,
        "peak_memory"      : max(peaks) if peaks else None,
    }


def _profile_fit(fn: Callable[[Tensor], Any], x: Tensor, config: BenchmarkConfig,
                 device: torch.device) -> dict[str, Any]:
    flops, times, peaks = _run_fits(fn, x, config, device)
    return {"flops": flops, "macs": None if flops is None else flops / 2, **_stats(times, peaks)}


# --- Real images by resolution ---

def _load_images(config: BenchmarkConfig) -> dict[tuple[int, int], list[Tensor]]:
    """``(H, W) → [1, 3, H, W]`` float images of ``config.data``, in ascending pixel count."""
    data = str(config.data)
    path = Path(data).expanduser()
    if path.exists():
        dataset: Any = ImageFolder(path)
    elif data in DATASETS:
        dataset = DATASETS.build(data, config.split)
    else:
        raise FileNotFoundError(f"data {data!r} is neither an existing path nor a dataset id")
    wanted = None if config.sizes is None else {s.lower() for s in config.sizes}
    groups: dict[tuple[int, int], list[Tensor]] = {}
    for i in range(len(dataset)):
        image = dataset[i]["image"]
        h, w = int(image.shape[-2]), int(image.shape[-1])
        if wanted is None or f"{w}x{h}" in wanted:
            groups.setdefault((h, w), []).append(image)
    if not groups:
        raise ValueError(f"no images in {data!r}" + ("" if wanted is None else f" with sizes {sorted(wanted)}"))
    return {k: groups[k] for k in sorted(groups, key=lambda hw: (hw[0] * hw[1], hw))}


def _measure_size(model: Callable[[Tensor], Any], images: list[Tensor], config: BenchmarkConfig, multiple: int,
                  device: torch.device) -> dict[str, Any]:
    """One resolution's row: ``model`` is the network, or the fit function when ``config.zero_shot``."""
    assert config.warmup is not None and config.runs is not None
    # One image on the device at a time: a peak must not include the size's other inputs.
    # scale=True: uint8 / 255, uint16 (16-bit PNG/TIFF) / 65535.
    batches = [pad_to_multiple(to_dtype(img, torch.float32, scale=True)[None]
                               .expand(config.batch_size, -1, -1, -1), multiple)[0]
               for img in images]
    if config.zero_shot:
        return _measure_fit_size(model, batches, config, device)
    flops = [count_flops(model, x.to(device)) for x in batches]
    times: list[float] = []
    peaks: list[int] = []
    for i, x_cpu in enumerate(batches):
        x = x_cpu.to(device)
        times += measure_latency(model, x, warmup=config.warmup if i == 0 else 0, runs=config.runs) \
            if config.latency else []
        peak = measure_peak_memory(model, x, device) if config.memory else None
        if peak is not None:
            peaks.append(peak)
    return {"macs": statistics.fmean(flops) / 2, "macs_min": min(flops) / 2, "macs_max": max(flops) / 2,
            **_stats(times, peaks)}


def _measure_fit_size(fit: Callable[[Tensor], Any], batches: list[Tensor], config: BenchmarkConfig,
                      device: torch.device) -> dict[str, Any]:
    """Zero-shot row: MACs counted on the first image only (counting costs a whole fit), warm-up there too."""
    flops, times, peaks = _run_fits(fit, batches[0].to(device), config, device)
    for x in batches[1:]:
        _, t, p = _run_fits(fit, x.to(device), config, device, count=False, warmup=0)
        times += t
        peaks += p
    macs = None if flops is None else flops / 2
    return {"macs": macs, "macs_min": macs, "macs_max": macs, **_stats(times, peaks)}


def _benchmark_data(config: BenchmarkConfig, model: nn.Module, fn: Callable[[Tensor], Any], multiple: int,
                    device: torch.device, seed: int, output: Path | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for (h, w), images in _load_images(config).items():
        row: dict[str, Any] = {"size": f"{w}x{h}", "images": len(images),
                               "input": [config.batch_size, 3, math.ceil(h / multiple) * multiple,
                                         math.ceil(w / multiple) * multiple]}
        try:
            row |= _measure_size(fn, images, config, multiple, device) | {"status": "ok"}
        except torch.OutOfMemoryError:
            row["status"] = "oom"
        except RuntimeError as err:  # e.g. a model's size limits; recorded, the next size still runs
            row["status"] = f"error: {str(err).splitlines()[0][:200]}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("%s %s: %s", config.model, row["size"], row["status"])
        rows.append(row)
    result: dict[str, Any] = {"model": config.model, "params": count_parameters(model), "rows": rows,
                              "config": {**asdict(config), "seed": seed, "device": str(device)}}
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
    console.print(_table(result))
    if output is not None:
        save_yaml(result, output)
        _write_csv(output.with_suffix(".csv"), rows)
        logger.info("wrote %s and %s", output, output.with_suffix(".csv"))
    return result


_COLUMNS = ("size", "images", "macs", "macs_min", "macs_max", "latency_ms", "latency_median_ms", "latency_std_ms",
            "peak_memory", "status")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_COLUMNS)
    writer.writerows([[("" if row.get(c) is None else row.get(c)) for c in _COLUMNS] for row in rows])
    _atomic_write_text(path, buf.getvalue())


def _table(result: Mapping[str, Any]) -> Table:
    device = result.get("device_name", result["config"]["device"])
    table = Table(title=f"{result['model']}: {result['params']:,} params on {device}")
    for name in ("Size", "Images", "GMACs", "Latency (ms)", "Median (ms)", "FPS", "Peak (MiB)", "Status"):
        table.add_column(name, justify="left" if name in ("Size", "Status") else "right", no_wrap=name == "Size")
    for row in result["rows"]:
        ok = row["status"] == "ok"
        lat, med, peak = row.get("latency_ms"), row.get("latency_median_ms"), row.get("peak_memory")
        table.add_row(
            row["size"], str(row["images"]),
            f"{row['macs'] / 1e9:.3f}" if ok else "-",
            f"{lat:.2f}" if ok and lat is not None else "-",
            f"{med:.2f}" if ok and med is not None else "-",
            f"{1000 * result['config']['batch_size'] / med:.1f}" if ok and med else "-",
            f"{peak / 2**20:.0f}" if ok and peak is not None else "-",
            row["status"],
        )
    return table


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
