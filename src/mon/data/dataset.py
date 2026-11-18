"""Folder-based datasets declared by their on-disk layout.

A dataset subclass states where each modality lives per split; files are paired across
modalities by relative path without extension. Datasets are stored in the normalized layout of
their task family (workspace ``data/README.md``), e.g. ``paired_layout`` for paired restoration.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> @register_dataset("my_lle", tasks={Task.LLE})  # doctest: +SKIP
    ... class MyLLE(FolderDataset):
    ...     folder = "my_lle"
    ...     layout = {Split.TEST: {"image": Modality("low"), "target": Modality("high")}}
"""

__all__ = [
    "DATASETS",
    "ClassFolderDataset",
    "FolderDataset",
    "ImageFolder",
    "Modality",
    "ModalityKind",
    "classification_layout",
    "paired_layout",
    "register_dataset",
    "unpaired_layout",
]

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors

from mon.core.console import get_logger
from mon.core.constants import IMAGE_EXTENSIONS, RAW_IMAGE_EXTENSIONS
from mon.core.enums import Split, Task
from mon.core.path import Path, datasets_dir
from mon.core.registry import Registry
from mon.core.size import Size
from mon.data.io import ImageMode, read_image
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

type DatasetClass = type["FolderDataset"] | type["ClassFolderDataset"]

DATASETS: Registry[DatasetClass] = Registry("datasets", autoload="mon.data.zoo")
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


def _is_hidden(path: Path, directory: Path) -> bool:
    """Whether any part of ``path`` below ``directory`` starts with ``.`` (``._x.png``, ``.ipynb_checkpoints/``)."""
    return any(part.startswith(".") for part in path.relative_to(directory).parts)


def _index_dir(directory: Path, exts: frozenset[str], *, recursive: bool = True) -> dict[str, Path]:
    """Map relative path without suffix (posix) → file, for files with an accepted suffix.

    Hidden files and folders (name starts with ``.``) are skipped: macOS writes ``._name``
    AppleDouble files next to images on external and network drives.
    """
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*") if recursive else directory.iterdir()):
        if path.suffix.lower() not in exts or _is_hidden(path, directory) or not path.is_file():
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
        folder: Dataset root relative to ``datasets_dir()``: the dataset id, in its normalized layout.
        layout: Split → modality name → ``Modality``. The first modality of each split is the
            primary one: it must be an image kind, and it defines the samples.
        classes: Class list (required for ``"voc"`` modalities; used by models and metrics).

    Override ``pair_key`` when a secondary modality isn't stored under the primary file's
    relative stem (e.g. one target per folder of inputs).

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

    @classmethod
    def pair_key(cls, modality: str, key: str) -> str:
        """Key of ``modality``'s file for the primary file ``key`` (relative path without suffix).

        The default pairs by identical key; subclasses override it for other pairings.
        """
        return key

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

        pair = {
            m: {k: self.pair_key(m, k) for k in keys} if m != primary_name else {k: k for k in keys}
            for m in self.modalities
        }
        missing: dict[str, list[str]] = {
            mod_name: [k for k in keys if pair[mod_name][k] not in indexed[mod_name]]
            for mod_name, mod in self.modalities.items() if mod.required and mod_name != primary_name
        }
        incomplete = {k for ks in missing.values() for k in ks}
        if incomplete:
            detail = "; ".join(f"{m}: {len(ks)} missing (e.g. {ks[:3]})" for m, ks in missing.items() if ks)
            if on_missing == "error":
                raise ValueError(f"{name} [{self.split.value}]: required files missing for {len(incomplete)} sample(s): {detail}")
            logger.warning("%s [%s]: skipping %d incomplete sample(s): %s", name, self.split.value, len(incomplete), detail)
        return [
            (k, {m: indexed[m][pair[m][k]] for m in self.modalities if pair[m][k] in indexed[m]})
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


class ClassFolderDataset(Dataset[Sample]):
    """Base for classification datasets stored as ``<split>/<class>/<image>``; subclasses declare the layout.

    Labels come from the parent folder; ids follow ``classes.txt`` at the dataset root (one
    name per line, in label-id order), not the alphabetical order of the folders. Samples are
    ``{"image", "label", "meta"}`` with ``label`` a 0-d ``int64`` tensor (``collate`` stacks it
    into ``[B]``) and ``meta.stem`` the path below the split folder (``"cat/00042"``).

    Class attributes to set in subclasses:
        folder: Dataset root relative to ``datasets_dir()`` (the dataset id).
        layout: Split → split folder relative to the root; see ``classification_layout``.
        image_mode: ``read_image`` mode, e.g. ``"gray"`` for MNIST.

    Args:
        split: Split to load; must be a key of ``layout``.
        root: Dataset root; defaults to ``datasets_dir() / folder``.
        transform: Called on each sample dict.

    Raises:
        ValueError: Bad split, a class folder missing from ``classes.txt`` (or a duplicate name in
            it), or no images.
        FileNotFoundError: Missing root, split folder, or ``classes.txt``.

    Examples:
        >>> import shutil, tempfile, torch
        >>> from mon.data.io import write_image
        >>> root = Path(tempfile.mkdtemp())
        >>> _ = (root / "classes.txt").write_text("zebra\\nant\\n")        # label ids: zebra=0, ant=1
        >>> for name in ("train/zebra/0.png", "train/ant/1.png"):
        ...     _ = write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), root / name)
        >>> class Animals(ClassFolderDataset):
        ...     layout = classification_layout("train")
        >>> ds = Animals("train", root=root)
        >>> ds.classes.names, ds.labels                     # sorted by path; ids from classes.txt
        (['zebra', 'ant'], [1, 0])
        >>> s = ds[0]
        >>> sorted(s), int(s["label"]), s["meta"].stem
        (['image', 'label', 'meta'], 1, 'ant/1')
        >>> shutil.rmtree(root)
    """

    folder    : ClassVar[str] = ""
    layout    : ClassVar[Mapping[Split, str]] = {}
    image_mode: ClassVar[ImageMode] = "rgb"
    classes_file: ClassVar[str] = "classes.txt"

    def __init__(
        self,
        split    : Split | str,
        *,
        root     : str | os.PathLike[str] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> None:
        name = type(self).__name__
        self.split = Split(split)
        if self.split not in self.layout:
            raise ValueError(f"{name} has no {self.split.value!r} split; available: {[s.value for s in self.layout]}")
        if root is not None:
            self.root = Path(root).expanduser()
        elif self.folder:
            self.root = datasets_dir() / self.folder
        else:
            raise ValueError(f"{name} defines no `folder`; pass root=")
        split_dir = self.root / self.layout[self.split]
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{name} split folder not found: {split_dir}")
        classes_path = self.root / self.classes_file
        if not classes_path.is_file():
            raise FileNotFoundError(f"{name}: {self.classes_file} not found in {self.root}")
        names = [line.strip() for line in classes_path.read_text().splitlines() if line.strip()]
        self.classes = ClassList.from_names(names)  # raises on duplicate names
        self.transform = transform

        folders = {p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}
        unknown = sorted(folders - set(names))
        if unknown:
            raise ValueError(f"{name} [{self.split.value}]: class folders not in {self.classes_file}: {unknown[:5]}")
        self._items: list[tuple[str, Path, int]] = []
        for class_def in self.classes:
            class_dir = split_dir / class_def.name
            if class_dir.is_dir():
                for key, path in _index_dir(class_dir, _READABLE_IMAGE_EXTS).items():
                    self._items.append((f"{class_def.name}/{key}", path, class_def.id))
        if not self._items:
            raise ValueError(f"{name} [{self.split.value}]: no images found in {split_dir}")
        self._items.sort(key=lambda item: item[0])

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Sample:
        stem, path, label = self._items[index]
        image  = read_image(path, self.image_mode)
        meta   = SampleMeta(
            index     = index,
            stem      = stem,
            split     = self.split,
            paths     = MappingProxyType({"image": path}),
            orig_size = Size.from_shape(image),
        )
        sample: Sample = {"image": image, "label": torch.tensor(label, dtype=torch.int64), "meta": meta}
        return self.transform(sample) if self.transform is not None else sample

    @property
    def labels(self) -> list[int]:
        """Label id of every sample, in index order (without reading images), e.g. for class-balanced samplers."""
        return [label for _, _, label in self._items]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(split={self.split.value!r}, root={str(self.root)!r}, {len(self)} samples)"

    @classmethod
    def splits(cls) -> frozenset[Split]:
        """Splits this dataset defines."""
        return frozenset(cls.layout)


class ImageFolder(Dataset[Sample]):
    """Images from any folder, file, or list of files: the input of prediction.

    Unlike ``FolderDataset`` there is no layout or registry entry; every image is one sample
    ``{"image", "meta"}`` with ``meta.split == Split.PREDICT``. ``meta.stem`` is the path
    relative to the folder without suffix (``"night/0001"``), so outputs can mirror the input
    tree: ``out_dir / f"{meta.stem}.png"``.

    Args:
        source: A folder, an image file, or a sequence of image files.
        mode: ``read_image`` mode: ``"rgb"`` (default), ``"gray"``, or ``"unchanged"``.
        transform: Called on each sample dict, as in ``FolderDataset``.
        recursive: Search subfolders of a folder ``source``.
        exts: Accepted suffixes for a folder ``source``; default: every image format
            ``read_image`` can decode (no camera RAW or HEIC).

    Raises:
        FileNotFoundError: If ``source`` (or a listed file) doesn't exist.
        ValueError: If no images are found, a listed file isn't an image, or two inputs share a stem
            (their outputs would overwrite each other).

    Examples:
        >>> import shutil, tempfile, torch
        >>> from mon.data.io import write_image
        >>> root = Path(tempfile.mkdtemp())
        >>> for name in ("b.png", "night/a.jpg", "._b.png"):
        ...     _ = write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), root / name)
        >>> ds = ImageFolder(root)
        >>> len(ds), [ds[i]["meta"].stem for i in range(len(ds))]  # sorted; hidden ._b.png skipped
        (2, ['b', 'night/a'])
        >>> ds[1]["image"].shape, ds[1]["meta"].split
        (torch.Size([3, 4, 6]), <Split.PREDICT: 'predict'>)
        >>> len(ImageFolder(root, recursive=False)), len(ImageFolder(root / "b.png"))
        (1, 1)
        >>> shutil.rmtree(root)
    """

    def __init__(
        self,
        source   : str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        *,
        mode     : ImageMode = "rgb",
        transform: Callable[[Sample], Sample] | None = None,
        recursive: bool = True,
        exts     : Iterable[str] | None = None,
    ) -> None:
        if mode not in ("rgb", "gray", "unchanged"):
            raise ValueError(f"mode must be 'rgb', 'gray', or 'unchanged', got {mode!r}")
        accepted = _READABLE_IMAGE_EXTS if exts is None else frozenset(
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts
        )
        self.mode      = mode
        self.transform = transform
        self.root: Path | None = None
        if isinstance(source, (str, os.PathLike)):
            path = Path(source).expanduser()
            if path.is_dir():
                self.root   = path
                self._items = list(_index_dir(path, accepted, recursive=recursive).items())
                if not self._items:
                    kind = "recursively " if recursive else ""
                    raise ValueError(f"no images ({', '.join(sorted(accepted))}) found {kind}in {path}")
                return
            files: list[Path] = [path]
        else:
            files = [Path(p).expanduser() for p in source]
            if not files:
                raise ValueError("source is an empty list of files")
        self._items = self._index_files(files)

    @staticmethod
    def _index_files(files: list[Path]) -> list[tuple[str, Path]]:
        items: dict[str, Path] = {}
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(f"image not found: {path}")
            if path.suffix.lower() not in _READABLE_IMAGE_EXTS:
                raise ValueError(f"not a readable image: {path} (accepted: {', '.join(sorted(_READABLE_IMAGE_EXTS))})")
            if path.stem in items:
                raise ValueError(f"two inputs share the stem {path.stem!r}: {items[path.stem]}, {path}; pass their folder instead")
            items[path.stem] = path
        return list(items.items())  # given order kept

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Sample:
        stem, path = self._items[index]
        image  = read_image(path, self.mode)
        meta   = SampleMeta(
            index     = index,
            stem      = stem,
            split     = Split.PREDICT,
            paths     = MappingProxyType({"image": path}),
            orig_size = Size.from_shape(image),
        )
        sample: Sample = {"image": image, "meta": meta}
        return self.transform(sample) if self.transform is not None else sample

    def __repr__(self) -> str:
        where = str(self.root) if self.root is not None else f"{len(self)} files"
        return f"{type(self).__name__}({where!r}, {len(self)} images)"


def register_dataset[C: DatasetClass](name: str, *, tasks: Iterable[Task]) -> Callable[[C], C]:
    """Register a ``FolderDataset``/``ClassFolderDataset`` subclass in ``DATASETS`` with ``tasks`` and its splits.

    Deriving ``splits`` from ``layout`` keeps registry filters (``DATASETS.names(splits=...)``)
    consistent with what the class can actually load. Registering never imports or touches data;
    ``DATASETS.build(name, split, ...)`` instantiates the class.

    Examples:
        >>> import shutil, tempfile, torch
        >>> from mon.data.io import write_image
        >>> root = Path(tempfile.mkdtemp())
        >>> for split in ("train", "test"):
        ...     for sub in ("images", "targets"):
        ...         _ = write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), root / split / sub / "a.png")
        >>> @register_dataset("doc_paired", tasks={Task.LLE})
        ... class DocPaired(FolderDataset):
        ...     layout = paired_layout("train", "test")
        >>> DATASETS.get("DOC_PAIRED") is DocPaired          # names are case-insensitive
        True
        >>> "doc_paired" in DATASETS.names(tasks=Task.LLE, splits=Split.TEST)
        True
        >>> ds = DATASETS.build("doc_paired", "test", root=root)  # = DocPaired("test", root=root)
        >>> len(ds), sorted(ds[0])
        (1, ['image', 'meta', 'target'])
        >>> DATASETS.get("doc_pairde")
        Traceback (most recent call last):
            ...
        mon.core.registry.RegistryKeyError: 'doc_pairde' is not registered in 'datasets'; did you mean 'doc_paired'?
        >>> _ = DATASETS.remove("doc_paired"); shutil.rmtree(root)
    """
    def decorator(cls: C) -> C:
        if not (isinstance(cls, type) and issubclass(cls, (FolderDataset, ClassFolderDataset))):
            raise TypeError(f"register_dataset expects a FolderDataset or ClassFolderDataset subclass, got {cls!r}")
        DATASETS.add(cls, name, tasks=frozenset(Task(t) for t in tasks), splits=cls.splits())
        return cls
    return decorator


# --- Normalized layouts ---

def paired_layout(*splits: Split | str) -> dict[Split, dict[str, Modality]]:
    """Layout for paired restoration datasets: ``<split>/images/`` → ``image``, ``<split>/targets/`` → ``target``.

    Examples:
        >>> layout = paired_layout("train", "test")
        >>> layout[Split.TEST]["target"].dir
        'test/targets'
    """
    if not splits:
        raise ValueError("paired_layout needs at least one split")
    parsed = [Split(s) for s in splits]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate splits: {[s.value for s in parsed]}")
    return {
        s: {"image": Modality(f"{s.value}/images"), "target": Modality(f"{s.value}/targets")}
        for s in parsed
    }


def unpaired_layout(*splits: Split | str) -> dict[Split, dict[str, Modality]]:
    """Layout for unpaired (no-reference) test sets: ``<split>/images/`` → ``image``, no targets.

    Examples:
        >>> unpaired_layout("test")[Split.TEST]["image"].dir
        'test/images'
    """
    if not splits:
        raise ValueError("unpaired_layout needs at least one split")
    parsed = [Split(s) for s in splits]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate splits: {[s.value for s in parsed]}")
    return {s: {"image": Modality(f"{s.value}/images")} for s in parsed}


def classification_layout(*splits: Split | str) -> dict[Split, str]:
    """Layout for classification datasets: split → folder of the same name holding ``<class>/<image>``.

    Examples:
        >>> classification_layout("train", "test")[Split.TEST]
        'test'
    """
    if not splits:
        raise ValueError("classification_layout needs at least one split")
    parsed = [Split(s) for s in splits]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate splits: {[s.value for s in parsed]}")
    return {s: s.value for s in parsed}
