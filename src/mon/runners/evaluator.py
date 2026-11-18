"""Score saved predictions against targets with image-quality metrics.

``evaluate(cfg)`` reads a folder of predictions (e.g. a ``predict`` run directory), pairs each with
its target by stem (the relative path without suffix, as ``predict`` writes it), and computes the
configured metrics per image: full-reference ones on ``(pred, target)``, no-reference ones on
``pred``. Scoring the saved 8-bit files is the protocol papers use. Without targets (``data`` unset,
or a test set without them, e.g. ``dicm``), only no-reference metrics run, on every prediction.

Writes ``metrics.csv`` (one row per image) and ``metrics.yaml`` (means, directions, the resolved
config) into ``output_dir`` (default: the prediction folder); with ``gt_mean``, ``metrics_gt_mean.*``.

``gt_mean`` rescales each prediction so its mean gray level equals the target's, then clips to
``[0, 1]``, before the full-reference metrics (no-reference metrics see the prediction as saved).
This is the "GT mean" protocol of LLFlow and of HVI-CIDNet's official evaluation; it removes global
brightness errors. (``mon_old``'s ``use_gt_mean`` only changed the progress label: a no-op.)
"""

__all__ = [
    "EvaluateConfig",
    "evaluate",
]

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from omegaconf import MISSING, DictConfig, OmegaConf
from torch import Tensor

from mon.core.console import get_logger, print_dict, progress_bar
from mon.core.fileio import _atomic_write_text, save_yaml
from mon.core.path import Path
from mon.core.runtime import select_device
from mon.data.dataset import DATASETS, FolderDataset, ImageFolder
from mon.data.io import read_image
from mon.metrics.base import Metric, rgb_to_luma
from mon.metrics.registry import METRICS

logger = get_logger(__name__)


@dataclass
class EvaluateConfig:
    """Settings for ``evaluate`` (also the schema of ``mon evaluate`` configs).

    Attributes:
        pred: Folder of predictions (searched recursively), e.g. a ``predict`` run directory.
        data: Targets: a ``DATASETS`` id (its ``target`` modality; ``split`` picks the split) or a
            folder of target images. ``None``: no targets, no-reference metrics only.
        split: Dataset split when ``data`` is a dataset id.
        metrics: ``METRICS`` names. Full-reference ones are skipped (with a warning) without targets.
        metric_args: Constructor arguments per metric, e.g. ``{"lpips": {"net": "vgg"}}``.
        gt_mean: Match each prediction's mean gray level to its target's first (see module docstring).
        device: ``select_device`` spec.
        output_dir: Where ``metrics.csv``/``metrics.yaml`` go; ``None``: ``pred``.
        exist_ok: Overwrite existing metric files there (otherwise that raises).
    """

    pred       : str = MISSING
    data       : str | None = None
    split      : str = "test"
    metrics    : list[str] = field(default_factory=lambda: ["psnr", "ssim", "lpips", "niqe"])
    metric_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    gt_mean    : bool = False
    device     : str = "auto"
    output_dir : str | None = None
    exist_ok   : bool = False


# --- Helpers ---

def _as_config(cfg: EvaluateConfig | DictConfig | Mapping[str, Any]) -> EvaluateConfig:
    merged = OmegaConf.merge(OmegaConf.structured(EvaluateConfig), cfg)
    if OmegaConf.is_missing(merged, "pred"):
        raise ValueError("EvaluateConfig.pred is required")
    obj = OmegaConf.to_object(merged)
    assert isinstance(obj, EvaluateConfig)
    if not obj.metrics:
        raise ValueError("EvaluateConfig.metrics is empty")
    unknown = sorted(set(obj.metric_args) - set(obj.metrics))
    if unknown:
        raise ValueError(f"metric_args for metrics not in metrics: {unknown}")
    return obj


def _targets(cfg: EvaluateConfig) -> dict[str, Path] | None:
    """Stem → target file, or ``None`` when there are no targets."""
    if cfg.data is None:
        return None
    path = Path(cfg.data).expanduser()
    if path.is_dir():
        return {stem: files["image"] for stem, files in ImageFolder(path).items()}
    if cfg.data not in DATASETS:
        raise FileNotFoundError(f"data {cfg.data!r} is neither a folder nor a dataset id "
                                f"(datasets: {', '.join(DATASETS.names())})")
    dataset = DATASETS.build(cfg.data, cfg.split)
    if not isinstance(dataset, FolderDataset) or "target" not in dataset.modalities:
        logger.warning("%s [%s] has no targets: no-reference metrics only", cfg.data, cfg.split)
        return None
    targets = {stem: files["target"] for stem, files in dataset.items() if "target" in files}
    return targets or None


def _gt_mean(pred: Tensor, target: Tensor) -> Tensor:
    """Scale ``pred`` so its mean luma equals ``target``'s, then clip to ``[0, 1]``."""
    mean_target = rgb_to_luma(target).mean(dim=(1, 2, 3), keepdim=True)
    mean_pred   = rgb_to_luma(pred).mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return (pred * (mean_target / mean_pred)).clamp(0, 1)


def _load(path: Path, device: torch.device) -> Tensor:
    image: Tensor = read_image(path).as_subclass(Tensor)
    return image[None].to(device, torch.float32) / 255


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    _atomic_write_text(path, buf.getvalue())


# --- Public API ---

def evaluate(cfg: EvaluateConfig | DictConfig | Mapping[str, Any]) -> dict[str, float]:
    """Score ``cfg.pred`` and write the metric files; return the mean of each metric that ran.

    Raises:
        ValueError: Unknown metric, a target without a prediction, or a pred/target size mismatch.
        FileNotFoundError: Missing ``pred`` folder, or ``data`` that is neither a folder nor a dataset id.
        FileExistsError: If the metric files exist and ``exist_ok`` is false.
    """
    config  = _as_config(cfg)
    for name in config.metrics:
        METRICS.entry(name)  # unknown names raise here, before any work
    device  = select_device(config.device)
    preds   = {stem: files["image"] for stem, files in ImageFolder(Path(config.pred).expanduser()).items()}
    targets = _targets(config)

    if targets is None:
        pairs: list[tuple[str, Path, Path | None]] = [(stem, path, None) for stem, path in preds.items()]
    else:
        missing = sorted(set(targets) - set(preds))
        if missing:
            shown = ", ".join(missing[:5]) + (", …" if len(missing) > 5 else "")
            raise ValueError(f"{len(missing)} target(s) have no prediction in {config.pred}: {shown}")
        extra = len(set(preds) - set(targets))
        if extra:
            logger.info("Ignoring %d prediction(s) without a target (e.g. save_extra outputs)", extra)
        pairs = [(stem, preds[stem], path) for stem, path in targets.items()]

    names = [n for n in config.metrics if targets is not None or METRICS.entry(n).meta["mode"] == "nr"]
    skipped = [n for n in config.metrics if n not in names]
    if skipped:
        logger.warning("No targets: skipping full-reference metrics %s", skipped)
    if not names:
        raise ValueError(f"nothing to compute: no targets and no no-reference metric in {config.metrics}")

    out_dir = Path(config.output_dir).expanduser() if config.output_dir else Path(config.pred).expanduser()
    stem    = "metrics_gt_mean" if config.gt_mean else "metrics"
    csv_path, yaml_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.yaml"
    if not config.exist_ok and (csv_path.exists() or yaml_path.exists()):
        raise FileExistsError(f"{csv_path.name}/{yaml_path.name} already exist in {out_dir}; pass exist_ok=true")

    metrics: dict[str, Metric] = {n: METRICS.build(n, **config.metric_args.get(n, {})).to(device) for n in names}
    rows: list[list[Any]] = []
    with torch.inference_mode(), progress_bar() as pbar:
        for key, pred_path, target_path in pbar.track(pairs, description="Evaluating"):
            pred   = _load(pred_path, device)
            target = _load(target_path, device) if target_path is not None else None
            if target is not None and pred.shape != target.shape:
                raise ValueError(f"{key}: prediction {tuple(pred.shape[-2:])} and target {tuple(target.shape[-2:])} "
                                 "differ in size; predictions must be cropped back, never resized")
            scored = _gt_mean(pred, target) if (config.gt_mean and target is not None) else pred
            row: list[Any] = [key]
            for name, metric in metrics.items():
                score = metric(scored, target) if metric.mode == "fr" else metric(pred)
                row.append(float(score[0]))
            rows.append(row)

    means = {n: sum(r[i + 1] for r in rows) / len(rows) for i, n in enumerate(names)}
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, ["stem", *names], rows)
    save_yaml({
        "count"        : len(rows),
        "mean"         : means,
        "higher_better": {n: METRICS.entry(n).meta["higher_better"] for n in names},
        "config"       : OmegaConf.to_container(OmegaConf.structured(config)),
    }, yaml_path, overwrite=True)
    print_dict(means, title=f"{len(rows)} images" + (" (GT mean)" if config.gt_mean else ""))
    logger.info("Wrote %s and %s", csv_path, yaml_path)
    return means

