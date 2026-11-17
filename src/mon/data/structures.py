"""Class definitions and per-sample metadata.

Image-like data uses ``torchvision.tv_tensors`` directly (``Image``, ``Mask``, ``Video``,
``BoundingBoxes``, ``KeyPoints``); this module adds what they lack: validated class lists that
map annotation ids to training ids, and the metadata carried in ``sample["meta"]``.
"""

__all__ = [
    "ClassDef",
    "ClassList",
    "Sample",
    "SampleMeta",
]

import colorsys
import difflib
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import torch

from mon.core.enums import Split
from mon.core.fileio import load_yaml, save_yaml
from mon.core.path import Path
from mon.core.size import Size
from mon.core.type_aliases import Int3

type Sample = dict[str, Any]
"""A dataset sample: modality name → ``tv_tensors`` value, plus ``"meta"`` → ``SampleMeta``."""

_GOLDEN = 0.618033988749895


def _default_color(class_id: int) -> Int3:
    """Deterministic, well-spread RGB color for ``class_id`` (golden-ratio hue steps)."""
    r, g, b = colorsys.hsv_to_rgb((class_id * _GOLDEN) % 1.0, 0.65, 0.95)
    return round(r * 255), round(g * 255), round(b * 255)


# --- Classes ---

@dataclass(frozen=True, slots=True)
class ClassDef:
    """One class.

    Attributes:
        name: Unique name within a ``ClassList``.
        id: Training id (``>= 0``), the value models predict.
        color: RGB in ``[0, 255]``; defaults to a deterministic color derived from ``id``.
        source_id: Id in the original annotation format (e.g. COCO's ``13`` for "stop sign"),
            used by ``ClassList.encode``.
    """

    name     : str
    id       : int
    color    : Int3 | None = None
    source_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"class name must be a non-empty string, got {self.name!r}")
        for attr in ("id", "source_id"):
            value = getattr(self, attr)
            if value is None and attr == "source_id":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{attr} of class {self.name!r} must be an int >= 0, got {value!r}")
        color = _default_color(self.id) if self.color is None else tuple(self.color)
        if len(color) != 3 or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in color):
            raise ValueError(f"color of class {self.name!r} must be 3 ints in [0, 255], got {self.color!r}")
        object.__setattr__(self, "color", color)


class ClassList:
    """Immutable, validated list of ``ClassDef`` ordered by ``id``.

    Look up by name with ``classes["car"]`` and by id with ``classes.by_id(3)``. Names, ids, and
    (non-``None``) source ids must be unique.

    Examples:
        >>> classes = ClassList.from_coco({1: "person", 3: "car", 13: "stop sign"})
        >>> classes["stop sign"].id, classes.encode([13, 1]).tolist()
        (2, [2, 0])
    """

    def __init__(self, classes: Iterable[ClassDef | Mapping[str, Any]] = ()) -> None:
        items = [c if isinstance(c, ClassDef) else ClassDef(**c) for c in classes]
        items.sort(key=lambda c: c.id)
        for attr in ("name", "id", "source_id"):
            seen: dict[Any, str] = {}
            for c in items:
                key = getattr(c, attr)
                if key is None:
                    continue
                if key in seen:
                    raise ValueError(f"duplicate {attr} {key!r} (classes {seen[key]!r} and {c.name!r})")
                seen[key] = c.name
        self._items  : tuple[ClassDef, ...] = tuple(items)
        self._by_name: dict[str, ClassDef] = {c.name: c for c in items}
        self._by_id  : dict[int, ClassDef] = {c.id: c for c in items}

    # --- Container ---

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ClassDef]:
        return iter(self._items)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, ClassDef):
            return self._by_name.get(item.name) == item
        return isinstance(item, str) and item in self._by_name

    def __getitem__(self, name: str) -> ClassDef:
        """Look up a class by name (use ``by_id`` for ids)."""
        if not isinstance(name, str):
            raise TypeError(f"look up classes by name (str); use by_id() for ids, got {name!r}")
        try:
            return self._by_name[name]
        except KeyError:
            close = difflib.get_close_matches(name, self._by_name, n=3, cutoff=0.6)
            hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            raise KeyError(f"no class named {name!r}{hint}") from None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClassList) and self._items == other._items

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        preview = ", ".join(c.name for c in self._items[:5]) + (", …" if len(self) > 5 else "")
        return f"ClassList({len(self)} classes: {preview})"

    def by_id(self, class_id: int) -> ClassDef:
        """Look up a class by training id."""
        try:
            return self._by_id[class_id]
        except KeyError:
            raise KeyError(f"no class with id {class_id}; ids are {self.ids}") from None

    # --- Views ---

    @property
    def names(self) -> list[str]:
        """Names in id order."""
        return [c.name for c in self._items]

    @property
    def ids(self) -> list[int]:
        """Training ids in ascending order."""
        return [c.id for c in self._items]

    @property
    def colors(self) -> list[Int3]:
        """RGB colors in id order."""
        return [c.color for c in self._items if c.color is not None]

    @property
    def num_classes(self) -> int:
        """Number of classes."""
        return len(self._items)

    @property
    def is_contiguous(self) -> bool:
        """Whether ids are exactly ``0 … N-1`` (what most losses and heads expect)."""
        return self.ids == list(range(len(self)))

    # --- Conversion ---

    def encode(self, source_ids: torch.Tensor | Sequence[int]) -> torch.Tensor:
        """Map annotation ids (``source_id``) to training ids, vectorized.

        Raises:
            ValueError: If some class has no ``source_id``.
            KeyError: If an input id matches no class's ``source_id``.
        """
        if any(c.source_id is None for c in self._items):
            raise ValueError("encode() needs every class to have a source_id")
        src = torch.as_tensor(source_ids, dtype=torch.int64)
        if src.numel() == 0:
            return src.clone()
        top = max(c.source_id or 0 for c in self._items)  # table sized by known ids, not by input values
        table = torch.full((top + 1,), -1, dtype=torch.int64)
        for c in self._items:
            table[c.source_id] = c.id
        in_range = (src >= 0) & (src <= top)
        out = torch.where(in_range, table[src.clamp(0, top)], torch.full_like(src, -1))
        unknown = out < 0
        if unknown.any():
            bad = sorted(set(src[unknown].tolist()))
            raise KeyError(f"source ids {bad} match no class")
        return out

    # --- Creation and I/O ---

    @classmethod
    def from_names(cls, names: Sequence[str], start: int = 0) -> Self:
        """Classes with ids ``start, start + 1, …`` in the given order (no ``source_id``)."""
        return cls(ClassDef(name=n, id=start + i) for i, n in enumerate(names))

    @classmethod
    def from_coco(cls, categories: Mapping[int, str], *, background: bool = False) -> Self:
        """Contiguous training ids from COCO-style categories, keeping the raw ids as ``source_id``.

        Args:
            categories: Raw category id → name (e.g. ``CocoAnnotations.categories``).
            background: Reserve id ``0`` for a ``"background"`` class (torchvision detection
                convention); real classes then start at ``1``.
        """
        start = 1 if background else 0
        items = [ClassDef(name=name, id=start + i, source_id=raw) for i, (raw, name) in enumerate(sorted(categories.items()))]
        if background:
            items.insert(0, ClassDef(name="background", id=0, color=(0, 0, 0)))
        return cls(items)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Self:
        """Load from YAML: ``classes: [{name, id, color?, source_id?}, …]``."""
        data = load_yaml(path)
        if not isinstance(data, dict) or not isinstance(data.get("classes"), list):
            raise ValueError(f"{path}: expected a mapping with a 'classes' list")
        return cls(data["classes"])

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = True) -> Path:
        """Atomically save as YAML (loadable with ``load``)."""
        rows = [{k: list(v) if k == "color" else v for k, v in asdict(c).items() if v is not None} for c in self._items]
        return save_yaml({"classes": rows}, path, overwrite=overwrite)


# --- Sample metadata ---

@dataclass(frozen=True)
class SampleMeta:
    """Metadata stored in ``sample["meta"]``; v2 transforms pass it through unchanged.

    Attributes:
        index: Position in the dataset.
        stem: File stem shared by the sample's files.
        split: Dataset split, if known.
        paths: Modality name → source file.
        orig_size: Image size before transforms, to map predictions back.
    """

    index    : int
    stem     : str
    split    : Split | None = None
    paths    : Mapping[str, Path] = field(default_factory=dict)
    orig_size: Size | None = None
