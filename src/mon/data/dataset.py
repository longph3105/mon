"""Folder-based datasets declared by their on-disk layout.

A dataset subclass states where each modality lives per split (in the dataset's upstream
folder layout); files are paired across modalities by relative path without extension.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> @register_dataset("my_lle", tasks={Task.LLE})  # doctest: +SKIP
    ... class MyLLE(FolderDataset):
    ...     folder = "enhance/MyLLE"
    ...     layout = {Split.TEST: {"image": Modality("low"), "target": Modality("high")}}
"""

__all__ = [
    "DATASETS",
    "FolderDataset",
    "Modality",
    "ModalityKind",
    "register_dataset",
]

import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from torch.utils.data import Dataset
from torchvision import tv_tensors

from mon.core.console import get_logger
from mon.core.constants import IMAGE_EXTENSIONS, RAW_IMAGE_EXTENSIONS
from mon.core.enums import Split, Task
from mon.core.path import Path, datasets_dir
from mon.core.registry import Registry
from mon.core.size import Size
from mon.data.io import read_image
from mon.data.labels import _empty_target, read_voc, read_yolo
from mon.data.structures import ClassList, Sample, SampleMeta

logger = get_logger(__name__)

type ModalityKind = Literal["image", "mask", "depth", "yolo", "voc"]

_IMAGE_KINDS = frozenset({"image", "mask", "depth"})
_BOX_KINDS   = frozenset({"yolo", "voc"})
_READABLE_IMAGE_EXTS = IMAGE_EXTENSIONS - RAW_IMAGE_EXTENSIONS - {".heic", ".heif"}
_DEFAULT_EXTS: dict[str, frozenset[str]] = {
    "image": _READABLE_IMAGE_EXTS,
    "mask" : _READABLE_IMAGE_EXTS,
    "depth": _READABLE_IMAGE_EXTS,
    "yolo" : frozenset({".txt"}),
    "voc"  : frozenset({".xml"}),
}

DATASETS: Registry[type["FolderDataset"]] = Registry("datasets", autoload="mon.data.zoo")
"""All registered datasets; the zoo is imported on first lookup. Metadata: ``tasks``, ``splits``."""


# --- Modality ---

@dataclass(frozen=True)
class Modality:
    """Where one modality's files live (relative to the dataset root) and how to read them.

    Attributes:
        dir: Directory relative to the dataset root; searched recursively.
        kind: ``"image"`` → ``Image`` (RGB); ``"mask"`` → ``Mask`` (class ids, nearest resize);
            ``"depth"`` → ``Image`` as stored (e.g. 16-bit); ``"yolo"``/``"voc"`` → ``boxes`` +
            ``labels`` (+ ``scores``/``difficult``).
        required: Every sample must have this file. Optional image modalities are left out of
            the sample when missing; optional box modalities yield empty boxes.
        exts: Accepted suffixes (lowercase, with dot); defaults depend on ``kind``.
    """

    dir     : str
    kind    : ModalityKind = "image"
    required: bool = True
    exts    : frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DEFAULT_EXTS:
            raise ValueError(f"unknown modality kind {self.kind!r}; expected one of {sorted(_DEFAULT_EXTS)}")
        exts = _DEFAULT_EXTS[self.kind] if self.exts is None else frozenset(
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.exts
        )
        object.__setattr__(self, "exts", exts)


def _index_dir(directory: Path, exts: frozenset[str]) -> dict[str, Path]:
    """Map relative path without suffix (posix) → file, for files with an accepted suffix."""
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in exts or not path.is_file():
            continue
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in files:
            raise ValueError(f"two files share the stem {key!r} in {directory}: {files[key].name}, {path.name}")
        files[key] = path
    return files


# --- Dataset ---

class FolderDataset(Dataset[Sample]):
    """Base for datasets stored as folders of files; subclasses only declare the layout.

    Class attributes to set in subclasses:
        folder: Dataset root relative to ``datasets_dir()`` (its upstream layout).
        layout: Split → modality name → ``Modality``. The first modality of each split is the
            primary one: it must be an image kind, and it defines the samples.
        classes: Class list (required for ``"voc"`` modalities; used by models and metrics).

    Args:
        split: Split to load; must be a key of ``layout``.
        root: Dataset root; defaults to ``datasets_dir() / folder``.
        transform: Called on each full sample dict (see ``mon.data.transforms``).
        on_missing: When a required file is missing for some primary file: ``"error"`` (default)
            raises listing them; ``"skip"`` drops those samples with one warning.

    Raises:
        ValueError: Bad split, layout, or options; missing required files (``"error"``); no samples.
        FileNotFoundError: Missing root or required modality directory.
    """

    folder : ClassVar[str] = ""
    layout : ClassVar[Mapping[Split, Mapping[str, Modality]]] = {}
    classes: ClassVar[ClassList | None] = None

    def __init__(
        self,
        split     : Split | str,
        *,
        root      : str | os.PathLike[str] | None = None,
        transform : Callable[[Sample], Sample] | None = None,
        on_missing: Literal["error", "skip"] = "error",
    ) -> None:
        name = type(self).__name__
        self.split = Split(split)
        if self.split not in self.layout:
            raise ValueError(f"{name} has no {self.split.value!r} split; available: {[s.value for s in self.layout]}")
        if on_missing not in ("error", "skip"):
            raise ValueError(f"on_missing must be 'error' or 'skip', got {on_missing!r}")
        self.modalities: Mapping[str, Modality] = MappingProxyType(dict(self.layout[self.split]))
        self._validate_layout()

        if root is not None:
            self.root = Path(root).expanduser()
        elif self.folder:
            self.root = datasets_dir() / self.folder
        else:
            raise ValueError(f"{name} defines no `folder`; pass root=")
        if not self.root.is_dir():
            raise FileNotFoundError(f"{name} root not found: {self.root}")
        self.transform = transform
        self._voc_ids  = {c.name: c.id for c in self.classes} if self.classes is not None else {}
        self._items    = self._build_index(on_missing)

    # --- Construction helpers ---

    def _validate_layout(self) -> None:
        name = type(self).__name__
        if not self.modalities:
            raise ValueError(f"{name} layout for {self.split.value!r} is empty")
        primary = next(iter(self.modalities.values()))
        if primary.kind not in _IMAGE_KINDS or not primary.required:
            raise ValueError(f"{name}: the first (primary) modality must be a required image kind, got {primary}")
        box_mods = [k for k, m in self.modalities.items() if m.kind in _BOX_KINDS]
        if len(box_mods) > 1:
            raise ValueError(f"{name}: at most one box modality per split, got {box_mods}")
        if any(self.modalities[k].kind == "voc" for k in box_mods) and self.classes is None:
            raise ValueError(f"{name}: 'voc' modalities need `classes` to map names to ids")
        reserved = {"meta", "boxes", "labels", "scores", "difficult", "iscrowd"} & set(self.modalities)
        if reserved:
            raise ValueError(f"{name}: modality names {sorted(reserved)} are reserved sample keys")

    def _build_index(self, on_missing: str) -> list[tuple[str, dict[str, Path]]]:
        name = type(self).__name__
        indexed: dict[str, dict[str, Path]] = {}
        for mod_name, mod in self.modalities.items():
            directory = self.root / mod.dir
            if not directory.is_dir():
                if mod.required:
                    raise FileNotFoundError(f"{name}: {mod_name!r} directory not found: {directory}")
                indexed[mod_name] = {}
                continue
            indexed[mod_name] = _index_dir(directory, mod.exts or frozenset())

        primary_name = next(iter(self.modalities))
        keys = list(indexed[primary_name])
        if not keys:
            raise ValueError(f"{name}: no {primary_name!r} files found in {self.root / self.modalities[primary_name].dir}")

        missing: dict[str, list[str]] = {
            mod_name: [k for k in keys if k not in indexed[mod_name]]
            for mod_name, mod in self.modalities.items() if mod.required and mod_name != primary_name
        }
        incomplete = {k for ks in missing.values() for k in ks}
        if incomplete:
            detail = "; ".join(f"{m}: {len(ks)} missing (e.g. {ks[:3]})" for m, ks in missing.items() if ks)
            if on_missing == "error":
                raise ValueError(f"{name} [{self.split.value}]: required files missing for {len(incomplete)} sample(s): {detail}")
            logger.warning("%s [%s]: skipping %d incomplete sample(s): %s", name, self.split.value, len(incomplete), detail)
        return [
            (k, {m: indexed[m][k] for m in self.modalities if k in indexed[m]})
            for k in keys if k not in incomplete
        ]

    # --- Dataset protocol ---

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Sample:
        stem, paths = self._items[index]
        sample: Sample = {}
        size: Size | None = None
        for mod_name, mod in self.modalities.items():
            path = paths.get(mod_name)
            if mod.kind in _IMAGE_KINDS:
                if path is None:
                    continue  # optional and missing
                if mod.kind == "image":
                    value: Any = read_image(path, "rgb")
                elif mod.kind == "mask":
                    value = tv_tensors.Mask(read_image(path, "gray")[0])
                else:
                    value = read_image(path, "unchanged")
                if size is None:
                    size = Size.from_shape(value)
                sample[mod_name] = value
            else:
                assert size is not None  # primary (image kind) comes first; checked in _validate_layout
                if path is None:  # optional and missing: no objects
                    target = _empty_target(size, *(("difficult",) if mod.kind == "voc" else ()))
                elif mod.kind == "yolo":
                    target = read_yolo(path, size)
                else:  # voc; the image's size wins over the XML's <size>
                    target = read_voc(path, self._voc_ids, canvas_size=size)
                sample.update(target)
        sample["meta"] = SampleMeta(index=index, stem=stem, split=self.split, paths=MappingProxyType(paths), orig_size=size)
        return self.transform(sample) if self.transform is not None else sample

    def __repr__(self) -> str:
        return f"{type(self).__name__}(split={self.split.value!r}, root={str(self.root)!r}, {len(self)} samples)"

    @classmethod
    def splits(cls) -> frozenset[Split]:
        """Splits this dataset defines."""
        return frozenset(cls.layout)


def register_dataset(name: str, *, tasks: Iterable[Task]) -> Callable[[type[FolderDataset]], type[FolderDataset]]:
    """Register a ``FolderDataset`` subclass in ``DATASETS`` with ``tasks`` and its ``layout``'s splits.

    Deriving ``splits`` from ``layout`` keeps registry filters (``DATASETS.names(splits=...)``)
    consistent with what the class can actually load.
    """
    def decorator(cls: type[FolderDataset]) -> type[FolderDataset]:
        if not (isinstance(cls, type) and issubclass(cls, FolderDataset)):
            raise TypeError(f"register_dataset expects a FolderDataset subclass, got {cls!r}")
        DATASETS.add(cls, name, tasks=frozenset(Task(t) for t in tasks), splits=cls.splits())
        return cls
    return decorator
