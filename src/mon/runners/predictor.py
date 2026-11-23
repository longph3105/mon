"""Run a model over images and save its outputs.

``predict(cfg)`` loads a registered model (or, for a zero-shot model, fits a fresh network to each
image), runs it on a folder, a file, a list of files, or a registered dataset split, and writes
each output, clamped to ``[0, 1]``, under the run directory, mirroring the input tree
(``meta.stem``). Inputs are padded to the model's ``input_multiple`` and outputs cropped back:
never resized. The run directory also gets ``config.yaml`` (resolved, with the seed used) and
``timing.yaml``.

Default run directory: ``runs_dir() / "predict" / <model>[-<weights>] / <data>``, where
``<data>`` is the folder name, the file stem, ``files`` for a list, or ``<dataset>_<split>``.

Examples:
    >>> cfg = PredictConfig(model="zero_dce", data="images/", weights="sice_me")  # doctest: +SKIP
    >>> predict(cfg)                                                             # doctest: +SKIP
    Path('.../runs/predict/zero_dce-sice_me/images')
"""

__all__ = [
    "PredictConfig",
    "predict",
]

import importlib
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass, field, replace
from typing import Any, cast, get_args

import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2

from mon.core.config import save_config
from mon.core.console import get_logger, progress_bar
from mon.core.enums import Task
from mon.core.fileio import save_yaml
from mon.core.path import Path, runs_dir
from mon.core.registry import Entry
from mon.core.runtime import seed_everything, select_device
from mon.core.timing import Profiler
from mon.core.weights import resolve_weights
from mon.data.dataset import DATASETS, ImageFolder
from mon.data.io import write_image
from mon.data.loader import create_dataloader
from mon.data.structures import Sample
from mon.data.transforms import PaddingMode, crop_to, pad_to_multiple
from mon.models.registry import MODELS, load
from mon.nn.tiling import tiled_forward

logger = get_logger(__name__)

DEFAULT_OUTPUT_KEYS: tuple[str, ...] = ("enhanced", "restored", "fused", "output")
"""Output saved when ``output_key`` is ``None``: the first of these the model returns."""


@dataclass
class PredictConfig:
    """Settings for ``predict`` (also the schema of ``mon predict`` configs).

    Attributes:
        model: ``MODELS`` name, e.g. ``"zero_dce"``.
        data: A folder (searched recursively), an image file, a list of image files, or a
            ``DATASETS`` id (``split`` picks the split). An existing path wins over an id.
        weights: For ``load``: ``"default"``, a member name, a checkpoint path, or ``None``
            (random init). Ignored when fitting a zero-shot model.
        model_args: Constructor arguments (e.g. ``{"scale_factor": 12}`` for Zero-DCE++).
        split: Dataset split when ``data`` is a dataset id.
        zero_shot: Fit a fresh network per image with the model's ``fit`` function. ``None``:
            the registry's ``zero_shot`` flag. ``False`` runs a zero-shot model's released weights.
        fit_args: Keyword arguments for the fit function (e.g. ``{"iterations": 500}``); its
            ``seed`` defaults to the run's seed.
        device: ``select_device`` spec: ``"auto"``, ``"cpu"``, ``"cuda:1"``, ``"mps"``.
        seed: Global seed (``None``: drawn, logged, and saved in ``config.yaml``).
        batch_size: Images per batch; above 1, all images must share one size.
        num_workers: ``DataLoader`` workers.
        pad_multiple: Pad ``H`` and ``W`` to a multiple of this (reflect) and crop outputs back.
            ``None``: the model's ``input_multiple`` (default 1). Must be a multiple of it.
        padding_mode: ``"reflect"``, ``"edge"``, ``"symmetric"``, or ``"constant"`` (zeros).
        tile: Run on overlapping ``tile``-sized tiles (``tiled_forward``) to bound memory;
            a multiple of ``pad_multiple``. Not for zero-shot fitting.
        tile_overlap: Pixels shared by neighboring tiles.
        output_key: Output to save; ``None``: the weights' ``meta["output_key"]`` if set (the
            official output for that checkpoint, e.g. RUAS ``upe``), else the first of
            ``DEFAULT_OUTPUT_KEYS`` present.
        save_extra: Also save the other image-shaped outputs (1 or 3 channels) as
            ``<stem>_<key><ext>``.
        ext: Output image format: ``".png"`` (8-bit), ``".jpg"``, ...
        run_dir: Output directory; ``None``: the default described in the module docstring.
        exist_ok: Write into an existing, non-empty run directory (files are overwritten).
            Otherwise that raises; nothing is ever deleted.
    """

    model       : str = MISSING
    data        : Any = MISSING
    weights     : str | None = "default"
    model_args  : dict[str, Any] = field(default_factory=dict)
    split       : str = "test"
    zero_shot   : bool | None = None
    fit_args    : dict[str, Any] = field(default_factory=dict)
    device      : str = "auto"
    seed        : int | None = None
    batch_size  : int = 1
    num_workers : int = 0
    pad_multiple: int | None = None
    padding_mode: str = "reflect"
    tile        : int | None = None
    tile_overlap: int = 64
    output_key  : str | None = None
    save_extra  : bool = False
    ext         : str = ".png"
    run_dir     : str | None = None
    exist_ok    : bool = False


# --- Helpers ---

def _as_config(cfg: PredictConfig | DictConfig | Mapping[str, Any]) -> PredictConfig:
    """Validate ``cfg`` against the schema and return a ``PredictConfig`` (missing fields raise)."""
    merged = OmegaConf.merge(OmegaConf.structured(PredictConfig), cfg)
    for key in ("model", "data"):
        if OmegaConf.is_missing(merged, key):
            raise ValueError(f"PredictConfig.{key} is required")
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, PredictConfig)
    if obj.padding_mode not in get_args(PaddingMode.__value__):
        raise ValueError(f"padding_mode must be one of {get_args(PaddingMode.__value__)}, got {obj.padding_mode!r}")
    if obj.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {obj.batch_size}")
    return obj


def _import_target(target: str) -> Callable[..., Tensor]:
    module, _, name = target.partition(":")
    return cast(Callable[..., Tensor], getattr(importlib.import_module(module), name))


def _is_path_like(data: Any) -> bool:
    return isinstance(data, (list, tuple)) or Path(str(data)).expanduser().exists()


def _build_inputs(cfg: PredictConfig) -> tuple[Dataset[Sample], str]:
    """The input dataset and its run-directory label."""
    transform = v2.ToDtype(torch.float32, scale=True)  # images only; uint8 -> [0, 1]
    data = cfg.data
    if _is_path_like(data):
        if isinstance(data, (list, tuple)):
            return ImageFolder([str(p) for p in data], transform=transform), "files"
        path = Path(str(data)).expanduser()
        return ImageFolder(path, transform=transform), path.name if path.is_dir() else path.stem
    name = str(data)
    if name not in DATASETS:
        raise FileNotFoundError(f"data {name!r} is neither an existing path nor a dataset id "
                                f"(datasets: {', '.join(DATASETS.names())})")
    return DATASETS.build(name, cfg.split, transform=transform), f"{name}_{cfg.split}"


def _weights_tag(weights: str | None) -> str | None:
    """Run-directory suffix for non-default weights."""
    if weights is None:
        return "random"
    if weights.strip().lower() == "default":
        return None
    path = Path(weights)
    return path.stem if path.suffix else weights.strip().lower()


def _resolve_run_dir(cfg: PredictConfig, data_label: str, zero_shot: bool) -> Path:
    if cfg.run_dir is not None:
        return Path(cfg.run_dir).expanduser()
    tag   = None if zero_shot else _weights_tag(cfg.weights)
    model = cfg.model if tag is None else f"{cfg.model}-{tag}"
    return runs_dir() / "predict" / model / data_label


def _input_multiple(cfg: PredictConfig, entry: Entry[Any]) -> int:
    required = int(getattr(entry.obj, "input_multiple", 1))
    if cfg.pad_multiple is None:
        return required
    if cfg.pad_multiple < 1 or cfg.pad_multiple % required:
        raise ValueError(f"pad_multiple={cfg.pad_multiple} must be a positive multiple of "
                         f"{cfg.model}'s input_multiple={required}")
    return cfg.pad_multiple


def _forward_fn(cfg: PredictConfig, entry: Entry[Any], zero_shot: bool, seed: int,
                device: torch.device, multiple: int) -> Callable[[Tensor], Tensor | Mapping[str, Tensor]]:
    """``x -> outputs`` for a padded ``[B, C, H, W]`` batch on ``device``."""
    if zero_shot:
        target = entry.meta.get("fit")
        if not target:
            raise ValueError(f"{cfg.model!r} has no 'fit' function in MODELS; set zero_shot=false")
        if cfg.tile is not None:
            raise ValueError("tile is not supported when fitting a zero-shot model (it fits whole images)")
        if cfg.weights is not None and cfg.weights.strip().lower() != "default":
            logger.warning("weights=%r ignored: %s fits a fresh network per image (zero_shot=false runs weights)",
                           cfg.weights, cfg.model)
        fit    = _import_target(str(target))
        kwargs = {"seed": seed, **cfg.fit_args}
        return lambda x: fit(x, **kwargs)

    if cfg.fit_args:
        logger.warning("fit_args ignored: %s runs its weights (zero_shot=false)", cfg.model)
    model = load(cfg.model, cfg.weights, device=device, **cfg.model_args)
    if cfg.tile is None:
        return cast(Callable[[Tensor], Tensor | Mapping[str, Tensor]], model)
    if cfg.tile % multiple:
        raise ValueError(f"tile={cfg.tile} must be a multiple of the pad multiple {multiple}")
    tile, overlap = cfg.tile, cfg.tile_overlap
    return lambda x: tiled_forward(model, x, tile=tile, overlap=overlap)


def _weights_output_key(cfg: PredictConfig, entry: Entry[Any], zero_shot: bool) -> str | None:
    """The chosen weights' ``meta["output_key"]`` (the checkpoint's official output), if any."""
    enum = getattr(entry.obj, "weights_enum", None)
    if zero_shot or enum is None:
        return None
    spec = resolve_weights(cfg.weights, enum)
    key  = spec.meta.get("output_key") if spec is not None else None
    return None if key is None else str(key)


def _output_key(outputs: Mapping[str, Tensor], wanted: str | None) -> str:
    if wanted is not None:
        if wanted not in outputs:
            raise KeyError(f"output_key {wanted!r} not in the model's outputs {sorted(outputs)}")
        return wanted
    for key in DEFAULT_OUTPUT_KEYS:
        if key in outputs:
            return key
    raise KeyError(f"none of {DEFAULT_OUTPUT_KEYS} in the model's outputs {sorted(outputs)}; set output_key")


def _is_image_like(t: Tensor, batch: int, h: int, w: int) -> bool:
    return t.ndim == 4 and t.shape[0] == batch and t.shape[1] in (1, 3) and t.shape[-2:] == (h, w)


def _check_run_dir(run_dir: Path, exist_ok: bool) -> None:
    if run_dir.exists() and not run_dir.is_dir():
        raise NotADirectoryError(f"run_dir {run_dir} exists and is not a directory")
    if run_dir.is_dir() and any(run_dir.iterdir()) and not exist_ok:
        raise FileExistsError(f"run_dir {run_dir} already has files; pass exist_ok=true to write into it, "
                              "or choose another run_dir")


# --- Public API ---

def predict(cfg: PredictConfig | DictConfig | Mapping[str, Any]) -> Path:
    """Run ``cfg.model`` over ``cfg.data`` and save the outputs; return the run directory.

    Raises:
        ValueError: For a missing ``model``/``data``, an exposure-stack (MEF) model, a zero-shot
            model without a fit function, or inconsistent ``pad_multiple``/``tile``.
        FileNotFoundError: If ``data`` is neither a path nor a dataset id.
        FileExistsError: If the run directory has files and ``exist_ok`` is false.
        KeyError: If ``output_key`` (or a default key) isn't among the model's outputs.
    """
    config = _as_config(cfg)
    entry  = MODELS.entry(config.model)
    tasks  = entry.meta.get("tasks", set())
    if Task.MEF in tasks:
        raise ValueError(f"{config.model!r} fuses exposure stacks ([B, N, C, H, W]); predict takes single images")
    zero_shot = bool(entry.meta.get("zero_shot", False)) if config.zero_shot is None else config.zero_shot

    seed    = seed_everything(config.seed)
    device  = select_device(config.device)
    dataset, label = _build_inputs(config)
    run_dir = _resolve_run_dir(config, label, zero_shot)
    _check_run_dir(run_dir, config.exist_ok)
    multiple = _input_multiple(config, entry)
    forward  = _forward_fn(config, entry, zero_shot, seed, device, multiple)
    wanted   = config.output_key or _weights_output_key(config, entry, zero_shot)

    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(replace(config, seed=seed, zero_shot=zero_shot, run_dir=str(run_dir)), run_dir / "config.yaml")
    mode = "fitting per image" if zero_shot else f"weights={config.weights!r}"
    count = len(cast(Sized, dataset))
    logger.info("Predicting %d images with %s (%s) on %s -> %s", count, config.model, mode, device, run_dir)

    loader   = create_dataloader(dataset, config.batch_size, num_workers=config.num_workers, seed=seed)
    profiler = Profiler(sync=device.type == "cuda")
    key: str | None = None
    with torch.inference_mode(), progress_bar() as pbar:
        for batch in pbar.track(loader, description="Predicting"):
            with profiler.stage("infer"):
                x       = batch["image"].to(device)
                x, size = pad_to_multiple(x, multiple, padding_mode=cast(PaddingMode, config.padding_mode))
                raw     = forward(x)
                outputs = {"output": raw} if isinstance(raw, Tensor) else dict(raw)
                key     = key or _output_key(outputs, wanted)
            with profiler.stage("save"):
                b, h, w = x.shape[0], x.shape[-2], x.shape[-1]
                saved   = {key: outputs[key]}
                if config.save_extra:
                    saved |= {k: v for k, v in outputs.items() if k != key and _is_image_like(v, b, h, w)}
                for name, value in saved.items():
                    images = crop_to(value, size).clamp(0, 1).float().cpu()
                    suffix = "" if name == key else f"_{name}"
                    for image, meta in zip(images, batch["meta"]):
                        write_image(image, run_dir / f"{meta.stem}{suffix}{config.ext}")

    save_yaml(profiler.summary(), run_dir / "timing.yaml")
    profiler.print_summary(f"{config.model} on {label}")
    logger.info("Saved %d images to %s", count, run_dir)
    return run_dir
