"""Train a registered model with its recipe, validate, checkpoint, and resume.

``train(cfg)`` resolves the config (``TrainConfig`` defaults ← the model's ``TrainRecipe.defaults``
← the user's values; see ``resolve_train_config``), builds the model (fresh, seeded, plus the
recipe's ``init``; or from ``weights`` to fine-tune), and runs a plain loop: the recipe's step
gives the loss, the runner does backward, AMP scaling, gradient clipping, the optimizer and
scheduler steps. Validation (optional) uses the evaluation protocol: inputs padded to the model's
``input_multiple``, outputs cropped back, clamped, and rounded to 8-bit levels, then ``mon.metrics``.

Run directory (default ``runs_dir() / "train" / <model> / <data> / <YYYY-MM-DD_HHMMSS>[_<name>]``):

- ``config.yaml``: the resolved config, with the seed;
- ``log.csv``: one row per epoch (step, lr, seconds, mean train terms, validation metrics);
- ``last.ckpt``: model, optimizer, scheduler, AMP scaler, epoch, step, best, RNG states, config
  (``resume=<run dir>`` continues exactly where it stopped);
- ``best.pt``: weights with the best ``monitor`` value; ``epoch_NNNN.pt`` every ``save_every`` epochs.
  Weight files load with ``mon.models.load(<model>, weights=<path>)``.

Training resizes (``resize``, or up to the model's ``input_multiple``); it never pads.
"""

__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainConfig",
    "resolve_train_config",
    "train",
]

import copy
import csv
import importlib
import io
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

from mon.core.config import save_config
from mon.core.console import get_logger, progress_bar
from mon.core.path import Path, runs_dir
from mon.core.runtime import seed_everything, select_device
from mon.data.dataset import DATASETS, ImageFolder
from mon.data.loader import collate, create_dataloader
from mon.data.structures import Sample
from mon.data.transforms import ClampPixels, ResizeToMultiple, crop_to, pad_to_multiple
from mon.metrics.base import Metric
from mon.metrics.registry import METRICS
from mon.models.registry import MODELS, load
from mon.nn.training import TrainRecipe
from mon.runners.predictor import _check_run_dir, _output_key

logger = get_logger(__name__)


@dataclass
class OptimizerConfig:
    """``name``: ``"adam"``, ``"adamw"``, or ``"sgd"`` (``momentum`` is SGD's)."""

    name        : str = "adam"
    lr          : float = 1e-3
    weight_decay: float = 0.0
    betas       : list[float] = field(default_factory=lambda: [0.9, 0.999])
    momentum    : float = 0.9


@dataclass
class SchedulerConfig:
    """Per-epoch LR schedule. ``name``: ``"none"``, ``"step"`` (``step_size``, ``gamma``),
    ``"multistep"`` (``milestones``, ``gamma``), or ``"cosine"`` (to ``eta_min`` over ``epochs``)."""

    name      : str = "none"
    step_size : int = 30
    milestones: list[int] = field(default_factory=list)
    gamma     : float = 0.1
    eta_min   : float = 0.0


@dataclass
class TrainConfig:
    """Settings for ``train`` (also the schema of ``mon train`` configs).

    Attributes:
        model: ``MODELS`` name with a ``train`` recipe (``MODELS.names(...)``; see ``mon.nn.training``).
        model_args: Constructor arguments.
        weights: Start from these weights (fine-tuning: ``load`` spec); ``None``: train from scratch.
        data: Training set: a ``DATASETS`` id (``split``) or a folder of images (unpaired training).
        split: Dataset split.
        resize: ``[H, W]`` every training image is resized to (bicubic, antialiased), rounded up to the
            model's ``input_multiple``; ``None``: native size, rounded up the same way. Pixels are then
            clamped to ``[0, 1]`` (bicubic overshoots).
        crop: Random square crop side after resizing; ``None``: none.
        hflip: Random horizontal flips.
        shuffle: Shuffle each epoch (order seeded by ``seed + epoch``, so resuming repeats it).
        cache: Decode and transform the training set once (with ``num_workers`` workers) and keep it in
            memory as float32; epochs then read RAM in the main process. The same tensors as without it,
            so only for deterministic transforms (no ``crop``/``hflip``). Memory: ``N·3·H·W·4`` bytes.
        cache_device: Where the cache lives: ``"cpu"`` (RAM), or ``"device"`` (the training device's
            memory, e.g. GPU VRAM: no host-to-device copy per batch).
        batch_size: Images per batch, or ``"auto"``: the largest power of two whose training step
            (forward + backward) peaks under ``AUTO_BATCH_MEMORY`` of the GPU, probed on a copy of the
            model before training (CUDA only; the result is saved in ``config.yaml``). Recipes keep
            their official batch size unless ``auto`` is asked for explicitly: it departs from the
            paper's recipe (the learning rate is not rescaled).
        num_workers: ``DataLoader`` workers.
        epochs: Training epochs.
        optimizer, scheduler: See ``OptimizerConfig``, ``SchedulerConfig``.
        grad_clip: Max gradient norm (``clip_grad_norm_``); ``None``: no clipping.
        eval_mode_after: Train in eval mode (batch norm uses its running statistics and stops
            updating them) after this many epochs; ``None``: train mode throughout. Only for recipes
            whose official script does this (Zero-IG: ``1``).
        amp: Mixed precision (float16 autocast + gradient scaling; CUDA only).
        seed: Global seed (``None``: drawn and saved). ``deterministic``: deterministic algorithms.
        device: ``select_device`` spec.
        val_data, val_split: Validation set (dataset id or folder); ``None``: no validation.
        val_every: Validate every this many epochs (and after the last one).
        val_metrics: ``METRICS`` names; full-reference ones need targets.
        output_key: Model output scored in validation; ``None``: ``enhanced``/``restored``/….
        monitor: What ``best.pt`` tracks: ``"loss"`` (train loss) or a validation metric name;
            ``None``: the first validation metric, else ``"loss"``.
        save_every: Also keep ``epoch_NNNN.pt`` every this many epochs (``0``: never).
        run_dir, name: Run directory, or a suffix for the default one.
        exist_ok: Write into an existing, non-empty ``run_dir``.
        resume: A run directory (or its ``last.ckpt``) to continue. Its saved config is used; only
            ``epochs``, ``device``, and ``num_workers`` may be changed. A ``cosine`` schedule then
            anneals over the new ``epochs``, from the LR it gives at the current epoch.
    """

    model       : str = MISSING
    model_args  : dict[str, Any] = field(default_factory=dict)
    weights     : str | None = None
    data        : str = MISSING
    split       : str = "train"
    resize      : list[int] | None = None
    crop        : int | None = None
    hflip       : bool = False
    shuffle     : bool = True
    cache       : bool = False
    cache_device: str = "cpu"
    batch_size  : Any = 1  # int, or "auto"
    num_workers : int = 4
    epochs      : int = 1
    optimizer   : OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler   : SchedulerConfig = field(default_factory=SchedulerConfig)
    grad_clip   : float | None = None
    eval_mode_after: int | None = None
    amp         : bool = False
    seed        : int | None = None
    deterministic: bool = False
    device      : str = "auto"
    val_data    : str | None = None
    val_split   : str = "test"
    val_every   : int = 1
    val_metrics : list[str] = field(default_factory=lambda: ["psnr", "ssim"])
    output_key  : str | None = None
    monitor     : str | None = None
    save_every  : int = 0
    run_dir     : str | None = None
    name        : str | None = None
    exist_ok    : bool = False
    resume      : str | None = None


_RESUME_KEYS = {"resume", "epochs", "device", "num_workers"}

AUTO_BATCH_MEMORY = 0.85
"""Fraction of GPU memory ``batch_size="auto"`` may use at the step's peak (the rest: optimizer state, slack)."""


# --- Config ---

def _recipe(model: str) -> TrainRecipe:
    entry  = MODELS.entry(model)
    target = entry.meta.get("train")
    if not target:
        trainable = [e.name for e in MODELS.entries() if e.meta.get("train")]
        raise ValueError(f"{model!r} has no training recipe; models with one: {trainable}")
    module, _, attr = str(target).partition(":")
    recipe = getattr(importlib.import_module(module), attr)
    assert isinstance(recipe, TrainRecipe), target
    return recipe


def _to_container(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, TrainConfig):
        cfg = OmegaConf.structured(cfg)
    if isinstance(cfg, DictConfig):
        return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def resolve_train_config(cfg: TrainConfig | DictConfig | Mapping[str, Any]) -> TrainConfig:
    """``TrainConfig`` defaults ← the model's recipe defaults ← ``cfg``; validated.

    Pass only the values you set (a dict or a schema-less ``DictConfig``): a full ``TrainConfig``
    instance carries every default and so overrides the whole recipe.

    Raises:
        ValueError: Missing ``model``/``data``, no recipe, or invalid values.
    """
    user = _to_container(cfg)
    if user.get("resume"):
        return _resume_config(user)
    if not user.get("model"):
        raise ValueError("TrainConfig.model is required")
    recipe = _recipe(str(user["model"]))
    merged = OmegaConf.merge(OmegaConf.structured(TrainConfig), dict(recipe.defaults), user)
    if OmegaConf.is_missing(merged, "data"):
        raise ValueError(f"TrainConfig.data is required ({user['model']}'s recipe names no training set)")
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, TrainConfig)
    _validate(obj)
    return obj


def _validate(cfg: TrainConfig) -> None:
    if not (cfg.batch_size == "auto" or (isinstance(cfg.batch_size, int) and cfg.batch_size >= 1)):
        raise ValueError(f"batch_size must be an int >= 1 or 'auto', got {cfg.batch_size!r}")
    if cfg.epochs < 1 or cfg.val_every < 1 or cfg.save_every < 0:
        raise ValueError("epochs and val_every must be >= 1, save_every >= 0")
    if cfg.optimizer.name not in ("adam", "adamw", "sgd"):
        raise ValueError(f"optimizer.name must be adam, adamw or sgd, got {cfg.optimizer.name!r}")
    if cfg.scheduler.name not in ("none", "step", "multistep", "cosine"):
        raise ValueError(f"scheduler.name must be none, step, multistep or cosine, got {cfg.scheduler.name!r}")
    if cfg.eval_mode_after is not None and cfg.eval_mode_after < 0:
        raise ValueError(f"eval_mode_after must be >= 0 or None, got {cfg.eval_mode_after}")
    if cfg.cache_device not in ("cpu", "device"):
        raise ValueError(f"cache_device must be 'cpu' or 'device', got {cfg.cache_device!r}")
    if cfg.cache and (cfg.crop is not None or cfg.hflip):
        raise ValueError("cache needs deterministic transforms: drop crop and hflip, or cache=false")
    if cfg.resize is not None and len(cfg.resize) != 2:
        raise ValueError(f"resize must be [H, W], got {cfg.resize}")
    for name in cfg.val_metrics:
        METRICS.entry(name)
    if cfg.monitor not in (None, "loss") and cfg.monitor not in cfg.val_metrics:
        raise ValueError(f"monitor must be 'loss' or one of val_metrics {cfg.val_metrics}, got {cfg.monitor!r}")
    if cfg.monitor not in (None, "loss") and cfg.val_data is None:
        raise ValueError(f"monitor={cfg.monitor!r} needs val_data")


def _resume_config(user: dict[str, Any]) -> TrainConfig:
    extra = sorted({k for k, v in user.items() if v is not None} - _RESUME_KEYS)
    if extra:
        allowed = sorted(_RESUME_KEYS - {"resume"})
        raise ValueError(f"resume uses the run's saved config; only {allowed} may change, got {extra}")
    run_dir = _resume_dir(str(user["resume"]))
    saved   = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(run_dir / "config.yaml"))
    changes = {k: v for k, v in user.items() if k != "resume" and v is not None}
    obj = OmegaConf.to_object(OmegaConf.merge(saved, changes, {"resume": str(run_dir), "run_dir": str(run_dir)}))
    assert isinstance(obj, TrainConfig)
    _validate(obj)
    return obj


def _resume_dir(path: str) -> Path:
    p = Path(path).expanduser()
    run_dir = p.parent if p.is_file() else p
    if not (run_dir / "last.ckpt").is_file() or not (run_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"{run_dir} has no last.ckpt and config.yaml to resume from")
    return run_dir


# --- Data ---

def _dataset(data: str, split: str, transform: v2.Transform) -> tuple[Dataset[Sample], str]:
    path = Path(data).expanduser()
    if path.is_dir():
        return ImageFolder(path, transform=transform), path.name
    if data not in DATASETS:
        known = ", ".join(DATASETS.names())
        raise FileNotFoundError(f"data {data!r} is neither a folder nor a dataset id (datasets: {known})")
    return DATASETS.build(data, split, transform=transform), f"{data}_{split}"


class _CachedDataset(Dataset[Sample]):
    """Every sample of ``dataset``, transformed once and kept in memory."""

    def __init__(self, dataset: Dataset[Sample], num_workers: int, device: torch.device | None = None) -> None:
        loader = create_dataloader(dataset, None, num_workers=num_workers, seed=0)  # single samples
        with progress_bar(transient=True) as pbar:
            # Copy (to ``device``, or a clone in RAM): tensors from workers live in shared memory, and holding
            # thousands of them exhausts file descriptors.
            self.samples: list[Sample] = [
                {k: (v.to(device) if device is not None else v.clone()) if isinstance(v, Tensor) else v
                 for k, v in sample.items()}
                for sample in pbar.track(loader, total=len(cast(Any, dataset)), description="Caching")
            ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        sample: Sample = self.samples[index]
        return sample


def _auto_batch_size(model: nn.Module, loss_fn: nn.Module, recipe: TrainRecipe, sample: Sample,
                     device: torch.device, amp: bool, limit: int) -> int:
    """Largest power-of-two batch whose training step peaks under ``AUTO_BATCH_MEMORY`` of the GPU."""
    total = torch.cuda.get_device_properties(device).total_memory
    probe = copy.deepcopy(model).train()
    best, size = 0, 1
    with torch.random.fork_rng(devices=[device]):
        while size <= limit:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                batch = _to_device(collate([sample] * size), device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                    out = recipe.step(probe, loss_fn, batch)
                loss = out["loss"] if recipe.optimizers is None else sum(out[f"loss_{n}"] for n in recipe.optimizers)
                loss.backward()  # type: ignore[union-attr]  # Tensor.backward is unannotated
                peak = torch.cuda.max_memory_allocated(device)
            except torch.cuda.OutOfMemoryError:
                break
            finally:
                probe.zero_grad(set_to_none=True)
                batch = out = loss = None  # type: ignore[assignment]
            logger.info("batch_size=auto: %d -> peak %.2f GiB of %.2f GiB", size, peak / 2**30, total / 2**30)
            if peak > AUTO_BATCH_MEMORY * total:
                break
            best, size = size, size * 2
    del probe
    torch.cuda.empty_cache()
    if best == 0:
        raise RuntimeError("batch_size=auto: even one image doesn't fit; use a smaller resize or crop")
    return best


def _train_transform(cfg: TrainConfig, multiple: int) -> v2.Transform:
    steps: list[Any] = [v2.ToDtype(torch.float32, scale=True)]
    if cfg.resize is not None or multiple > 1:
        steps.append(ResizeToMultiple(multiple, cfg.resize, interpolation=v2.InterpolationMode.BICUBIC))
        steps.append(ClampPixels())  # bicubic + antialias overshoots [0, 1]
    if cfg.crop is not None:
        steps.append(v2.RandomCrop(cfg.crop))
    if cfg.hflip:
        steps.append(v2.RandomHorizontalFlip())
    return v2.Compose(steps)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if isinstance(v, Tensor) else v for k, v in batch.items()}


# --- Optimization ---

def _param_groups(model: nn.Module, recipe: TrainRecipe) -> dict[str, list[nn.Parameter]]:
    """``{optimizer name: parameters}``; ``{"": all}`` for a single-optimizer recipe."""
    if recipe.optimizers is None:
        return {"": list(model.parameters())}
    groups: dict[str, list[nn.Parameter]] = {}
    for name, parts in recipe.optimizers.items():
        groups[name] = []
        for part in parts:
            sub = getattr(model, part, None)
            if not isinstance(sub, nn.Module):
                raise ValueError(f"recipe optimizer {name!r}: {type(model).__name__} has no submodule {part!r}")
            groups[name] += list(sub.parameters())
    ids = [id(p) for ps in groups.values() for p in ps]
    if len(ids) != len(set(ids)):
        raise ValueError(f"recipe optimizers share parameters: {dict(recipe.optimizers)}")
    left = sum(p.numel() for p in model.parameters() if id(p) not in set(ids))
    if left:
        logger.warning("%d parameters are in no recipe optimizer and stay fixed", left)
    return groups


def _set_trainable(groups: Mapping[str, list[nn.Parameter]], active: str) -> None:
    """Only ``active``'s parameters require grad (the others' gradients would be discarded anyway)."""
    for name, params in groups.items():
        for p in params:
            p.requires_grad_(name == active)


def _optim_states(optims: Mapping[str, torch.optim.Optimizer],
                  scheds: Mapping[str, torch.optim.lr_scheduler.LRScheduler | None], multi: bool) -> dict[str, Any]:
    """Checkpoint entries: one state dict each (single optimizer), or ``{name: state}`` per optimizer."""
    o = {n: opt.state_dict() for n, opt in optims.items()}
    s = {n: sch.state_dict() if sch is not None else None for n, sch in scheds.items()}
    return {"optimizer": o, "scheduler": s} if multi else {"optimizer": o[""], "scheduler": s[""]}


def _optimizer(cfg: OptimizerConfig, params: Any) -> torch.optim.Optimizer:
    if cfg.name == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    cls = torch.optim.Adam if cfg.name == "adam" else torch.optim.AdamW
    return cls(params, lr=cfg.lr, betas=(cfg.betas[0], cfg.betas[1]), weight_decay=cfg.weight_decay)


def _scheduler(
    cfg: SchedulerConfig, optimizer: torch.optim.Optimizer, epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, cfg.step_size, cfg.gamma)
    if cfg.name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.milestones, cfg.gamma)
    if cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, cfg.eta_min)
    return None


def _retarget_cosine(sched: torch.optim.lr_scheduler.LRScheduler, epochs: int) -> None:
    """After a resume with a new ``epochs``: anneal to ``eta_min`` over it, not the saved ``T_max``.

    ``load_state_dict`` restores the old ``T_max``, so without this a longer resume would keep the old
    horizon (and cosine past it, back up). The LR jumps to the new curve at the current epoch; later
    steps follow it. An unchanged ``epochs`` leaves the state untouched (exact resume).
    """
    if not isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR) or sched.T_max == epochs:
        return
    sched.T_max = epochs
    t = sched.last_epoch
    lrs = [sched.eta_min + (base - sched.eta_min) * (1 + math.cos(math.pi * t / epochs)) / 2 for base in sched.base_lrs]
    for group, lr in zip(sched.optimizer.param_groups, lrs, strict=True):
        group["lr"] = lr
    sched._last_lr = lrs  # what get_last_lr() reports


# --- Checkpoints ---

def _rng_state() -> dict[str, Any]:
    kind, keys, pos, has_gauss, gauss = np.random.get_state()
    state: dict[str, Any] = {
        "torch" : torch.get_rng_state(),
        "numpy" : [kind, torch.from_numpy(keys.copy()), int(pos), int(has_gauss), float(gauss)],
        "python": list(random.getstate()),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    # RNG states must be CPU ByteTensors; the checkpoint may have been loaded onto the training device.
    torch.set_rng_state(state["torch"].cpu())
    kind, keys, pos, has_gauss, gauss = state["numpy"]
    np.random.set_state((kind, keys.cpu().numpy(), pos, has_gauss, gauss))
    version, internal, gauss_next = state["python"]
    random.setstate((version, tuple(internal), gauss_next))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def _save(obj: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_csv(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    exists = path.is_file()
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    if not exists:
        writer.writeheader()
    writer.writerow(row)
    with path.open("a", encoding="utf-8") as f:
        f.write(buf.getvalue())


# --- Validation ---

@torch.inference_mode()
def _validate_epoch(model: nn.Module, loader: Any, metrics: Mapping[str, Metric], multiple: int,
                    output_key: str | None, device: torch.device) -> dict[str, float]:
    training = model.training
    model.eval()
    sums = dict.fromkeys(metrics, 0.0)
    count = 0
    key = output_key
    for batch in loader:
        x, size = pad_to_multiple(batch["image"].to(device), multiple)
        raw  = model(x)
        outs = {"output": raw} if isinstance(raw, Tensor) else dict(raw)
        key  = key or _output_key(outs, None)
        pred = (crop_to(outs[key], size).clamp(0, 1) * 255).round() / 255  # as saved 8-bit files
        target = batch.get("target")
        target = target.to(device) if isinstance(target, Tensor) else None
        for name, metric in metrics.items():
            score = metric(pred, target) if metric.mode == "fr" else metric(pred)
            sums[name] += float(score.sum())
        count += pred.shape[0]
    model.train(training)
    return {f"val/{n}": v / max(count, 1) for n, v in sums.items()}


# --- Public API ---

def train(cfg: TrainConfig | DictConfig | Mapping[str, Any]) -> Path:
    """Train per ``cfg`` (see ``resolve_train_config``); return the run directory.

    Raises:
        ValueError: Invalid config, or full-reference ``val_metrics`` on a validation set without targets.
        FileExistsError: Non-empty run directory without ``exist_ok`` (not when resuming).
        FloatingPointError: A non-finite loss (the last checkpoint stays as it was).
    """
    config  = resolve_train_config(cfg)
    recipe  = _recipe(config.model)
    entry   = MODELS.entry(config.model)
    resume  = config.resume is not None
    seed    = seed_everything(config.seed, deterministic=config.deterministic)
    config  = replace(config, seed=seed)
    device  = select_device(config.device)
    if config.amp and device.type != "cuda":
        raise ValueError("amp needs a CUDA device")
    multiple = int(getattr(entry.obj, "input_multiple", 1))

    train_set, label = _dataset(config.data, config.split, _train_transform(config, multiple))
    if resume:
        run_dir = Path(str(config.run_dir))
    elif config.run_dir is not None:
        run_dir = Path(config.run_dir).expanduser()
    else:
        stamp   = time.strftime("%Y-%m-%d_%H%M%S") + (f"_{config.name}" if config.name else "")
        run_dir = runs_dir() / "train" / config.model / label / stamp
    if not resume:
        _check_run_dir(run_dir, config.exist_ok)

    # Model: fine-tune from weights, or fresh (seeded above) plus the recipe's init.
    if config.weights is not None:
        model = load(config.model, config.weights, **config.model_args)
    else:
        model = entry.obj(**config.model_args)
        if recipe.init is not None:
            recipe.init(model)
    model    = model.to(device).train()
    loss_fn  = recipe.loss().to(device)
    groups   = _param_groups(model, recipe)
    optims   = {n: _optimizer(config.optimizer, ps) for n, ps in groups.items()}
    scheds   = {n: _scheduler(config.scheduler, o, config.epochs) for n, o in optims.items()}
    multi    = recipe.schedule is not None
    scaler   = torch.amp.GradScaler(device.type, enabled=config.amp)

    val_loader = None
    metrics: dict[str, Metric] = {}
    if config.val_data is not None:
        val_set, _ = _dataset(config.val_data, config.val_split, v2.ToDtype(torch.float32, scale=True))
        metrics    = {n: METRICS.build(n).to(device) for n in config.val_metrics}
        if any(m.mode == "fr" for m in metrics.values()) and "target" not in val_set[0]:
            raise ValueError(f"{config.val_data} [{config.val_split}] has no targets for {config.val_metrics}")
        val_loader = create_dataloader(val_set, 1, num_workers=config.num_workers, seed=seed)
    monitor = config.monitor or (config.val_metrics[0] if val_loader is not None else "loss")
    higher  = monitor != "loss" and bool(METRICS.entry(monitor).meta["higher_better"])

    start_epoch, step = 0, 0
    best: dict[str, Any] = {"name": monitor, "value": None, "epoch": None}
    if resume:
        ckpt = torch.load(run_dir / "last.ckpt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        saved_optims = ckpt["optimizer"] if multi else {"": ckpt["optimizer"]}
        saved_scheds = ckpt["scheduler"] if multi else {"": ckpt["scheduler"]}
        for n, o in optims.items():
            o.load_state_dict(saved_optims[n])
            s = scheds[n]
            if s is not None and saved_scheds[n] is not None:
                s.load_state_dict(saved_scheds[n])
                _retarget_cosine(s, config.epochs)
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, step, best = ckpt["epoch"], ckpt["step"], ckpt["best"]
        _set_rng_state(ckpt["rng"])
        logger.info("Resuming %s at epoch %d (step %d)", run_dir, start_epoch, step)
        if start_epoch >= config.epochs:
            logger.info("Already trained %d of %d epochs: nothing to do", start_epoch, config.epochs)
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")

    workers, pin = config.num_workers, None
    if config.cache:
        on_device = config.cache_device == "device" and device.type != "cpu"
        train_set = _CachedDataset(train_set, config.num_workers, device if on_device else None)
        workers   = 0  # in-memory samples: workers would only copy them
        pin       = False if on_device else None  # device tensors can't be pinned
        logger.info("Cached %d training samples on %s", len(train_set), device if on_device else "cpu")
    if config.batch_size == "auto":
        if device.type != "cuda":
            raise ValueError("batch_size='auto' needs a CUDA device")
        if config.resize is None and config.crop is None:
            logger.warning("batch_size=auto probes the first image's size; images of other sizes can't share a batch")
        chosen = _auto_batch_size(model, loss_fn, recipe, train_set[0], device, config.amp, len(cast(Any, train_set)))
        official = recipe.defaults.get("batch_size")
        if official is not None and official != chosen:
            logger.warning("batch_size=auto chose %d; the official %s recipe uses %d (learning rate not rescaled): "
                           "this run does not reproduce the paper", chosen, config.model, official)
        config = replace(config, batch_size=chosen)
        save_config(config, run_dir / "config.yaml")
    loader = create_dataloader(train_set, config.batch_size, shuffle=config.shuffle, seed=seed,
                               num_workers=workers, drop_last=False, pin_memory=pin)
    columns: list[str] | None = None
    logger.info("Training %s on %s (%d images) for %d epochs on %s -> %s", config.model, label,
                len(cast(Any, train_set)), config.epochs, device, run_dir)

    with progress_bar() as pbar:
        task = pbar.add_task("Training", total=config.epochs, completed=start_epoch)
        for epoch in range(start_epoch, config.epochs):
            if loader.generator is not None:
                loader.generator.manual_seed(seed + epoch)  # same order and worker seeds after a resume
            t0, sums, batches = time.perf_counter(), dict[str, float](), 0
            model.train(config.eval_mode_after is None or epoch < config.eval_mode_after)
            for batch in loader:
                active = recipe.schedule(step + 1) if recipe.schedule is not None else ""
                if active is None:  # a scheduled step without an update: the batch is consumed
                    step += 1
                    continue
                if multi:
                    _set_trainable(groups, active)
                optim = optims[active]
                batch = _to_device(batch, device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=config.amp):
                    out = recipe.step(model, loss_fn, batch)
                if multi:
                    out = {"loss": out[f"loss_{active}"], **out}
                loss = out["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss {loss.item()} at epoch {epoch + 1}, step {step + 1}")
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]  # Tensor.backward is unannotated
                if config.grad_clip is not None:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(groups[active], config.grad_clip)
                scaler.step(optim)
                scaler.update()
                step += 1
                batches += 1
                for k, v in out.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
            lr = next(iter(optims.values())).param_groups[0]["lr"]
            for sched in scheds.values():
                if sched is not None:
                    sched.step()
            seconds = round(time.perf_counter() - t0, 3)
            row: dict[str, Any] = {"epoch": epoch + 1, "step": step, "lr": lr, "seconds": seconds}
            row |= {f"train/{k}": v / max(batches, 1) for k, v in sums.items()}
            last = epoch + 1 == config.epochs
            if val_loader is not None and ((epoch + 1) % config.val_every == 0 or last):
                row |= _validate_epoch(model, val_loader, metrics, multiple, config.output_key, device)

            value = row.get("train/loss" if monitor == "loss" else f"val/{monitor}")  # none: no step updated
            if value is not None and (
                best["value"] is None or (value > best["value"] if higher else value < best["value"])
            ):
                best = {"name": monitor, "value": float(value), "epoch": epoch + 1}
                _save(model.state_dict(), run_dir / "best.pt")
            if config.save_every and (epoch + 1) % config.save_every == 0:
                _save(model.state_dict(), run_dir / f"epoch_{epoch + 1:04d}.pt")
            _save({
                "model": model.state_dict(), **_optim_states(optims, scheds, multi), "scaler": scaler.state_dict(),
                "epoch": epoch + 1, "step": step, "best": best, "rng": _rng_state(),
                "config": OmegaConf.to_container(OmegaConf.structured(config)),
            }, run_dir / "last.ckpt")

            if columns is None and sums:  # the first epoch with an update fixes the columns
                train_cols = [f"train/{k}" for k in sums]
                val_cols   = [f"val/{n}" for n in metrics]
                columns    = ["epoch", "step", "lr", "seconds", *train_cols, *val_cols]
            if columns is not None:
                _append_csv(run_dir / "log.csv", row, columns)
            shown = ", ".join(f"{k.split('/', 1)[1]} {v:.4g}" for k, v in row.items() if "/" in k and v is not None)
            logger.info("epoch %d/%d: %s (%.1fs)", epoch + 1, config.epochs, shown, row["seconds"])
            pbar.update(task, advance=1)

    if best["value"] is not None and math.isfinite(best["value"]):
        logger.info("Best %s %.6g at epoch %d -> %s", best["name"], best["value"], best["epoch"], run_dir / "best.pt")
    return run_dir
