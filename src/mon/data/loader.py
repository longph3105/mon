"""Batching for dict samples and reproducible DataLoader construction.

``collate`` stacks same-shape tensors (images, masks) and keeps per-object targets
(``boxes``, ``labels``, …) and ``meta`` as lists, the format torchvision detection models use.
"""

__all__ = [
    "LIST_KEYS",
    "collate",
    "create_dataloader",
]

from collections.abc import Callable, Sequence
from typing import Any

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from mon.data.structures import Sample

LIST_KEYS: frozenset[str] = frozenset({"boxes", "labels", "scores", "difficult", "iscrowd", "keypoints", "meta"})
"""Sample keys kept as per-sample lists by ``collate`` (their sizes differ between samples)."""


def collate(samples: Sequence[Sample]) -> dict[str, Any]:
    """Batch a list of sample dicts.

    Tensors under keys not in ``LIST_KEYS`` are stacked into ``[B, …]`` plain tensors;
    ``BoundingBoxes``/``KeyPoints``, per-object targets, ``meta``, and non-tensors become lists.

    Raises:
        ValueError: If samples have different keys, or tensors to stack differ in shape.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    keys = list(samples[0])
    for i, s in enumerate(samples[1:], start=1):
        if set(s) != set(keys):
            raise ValueError(f"sample {i} has keys {sorted(s)}, sample 0 has {sorted(keys)}; optional modalities "
                             "must be present in every sample of a batch (or use batch_size=1)")
    batch: dict[str, Any] = {}
    for key in keys:
        values = [s[key] for s in samples]
        first = values[0]
        ragged = isinstance(first, (tv_tensors.BoundingBoxes, tv_tensors.KeyPoints))
        if key in LIST_KEYS or not isinstance(first, torch.Tensor) or ragged:
            batch[key] = values
            continue
        shapes = {tuple(v.shape) for v in values}
        if len(shapes) > 1:
            raise ValueError(f"cannot stack {key!r}: shapes differ {sorted(shapes)}; resize or crop to one size in "
                             "the transforms, or use batch_size=1")
        batch[key] = torch.stack([v.as_subclass(torch.Tensor) for v in values])
    return batch


def _identity(sample: Any) -> Any:
    return sample


def _worker_init(worker_id: int) -> None:
    """Per-worker setup: stop OpenCV's own thread pool from oversubscribing the CPU.

    PyTorch already seeds ``random``, NumPy, and torch differently in each worker (from the
    loader's generator), so no extra seeding is needed here.
    """
    cv2.setNumThreads(0)


def create_dataloader(
    dataset           : Dataset[Sample],
    batch_size        : int | None = 1,
    *,
    shuffle           : bool = False,
    seed              : int | None = None,
    num_workers       : int = 0,
    drop_last         : bool = False,
    pin_memory        : bool | None = None,
    persistent_workers: bool | None = None,
    collate_fn        : Callable[[Sequence[Sample]], Any] = collate,
    **kwargs          : Any,
) -> DataLoader[Sample]:
    """Build a ``DataLoader`` whose shuffle order and worker RNG streams are reproducible.

    Args:
        dataset: Map-style dataset of sample dicts.
        batch_size: Samples per batch; ``None`` yields single samples (``collate`` is then skipped).
        shuffle: Shuffle each epoch.
        seed: Seed for the loader's generator (shuffle order and the base of each worker's
            seeds). ``None`` draws one from torch's global RNG, so runs seeded with
            ``seed_everything`` stay reproducible.
        num_workers: Worker processes (``0`` loads in the main process).
        drop_last: Drop the last incomplete batch.
        pin_memory: Pin host memory for faster GPU copies; ``None`` → whether CUDA is available.
        persistent_workers: Keep workers between epochs; ``None`` → ``num_workers > 0``.
        collate_fn: Batching function; defaults to ``collate``.
        **kwargs: Passed to ``DataLoader`` (e.g. ``prefetch_factor``, ``multiprocessing_context``).
    """
    if seed is None:
        seed = int(torch.randint(0, 2**62, (1,)).item())
    if batch_size is None and collate_fn is collate:
        collate_fn = _identity  # DataLoader passes single samples to collate_fn when not batching
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle,
        num_workers        = num_workers,
        drop_last          = drop_last,
        collate_fn         = collate_fn,
        pin_memory         = torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers = (num_workers > 0) if persistent_workers is None else persistent_workers,
        worker_init_fn     = _worker_init,
        generator          = generator,
        **kwargs,
    )
