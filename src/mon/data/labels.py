"""Read and write YOLO, VOC, and COCO box labels.

Every reader returns a sample-ready dict: ``"boxes"`` (``tv_tensors.BoundingBoxes``, absolute
``XYXY`` pixels, float32) and ``"labels"`` (int64 class ids), plus format-specific extras.
A missing label file can mean "no objects" (``missing_ok=True``, the YOLO convention); a
malformed one always raises, with the file and line, so bad labels never become silent negatives.

References:
    - YOLO format: https://docs.ultralytics.com/datasets/detect/
    - Pascal VOC devkit: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html
    - COCO format: https://cocodataset.org/#format-data
"""

__all__ = [
    "CocoAnnotations",
    "read_coco",
    "read_voc",
    "read_yolo",
    "write_yolo",
]

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision import tv_tensors

from mon.core.console import get_logger
from mon.core.fileio import _atomic_write_text, _prepare_target, load_json
from mon.core.path import Path
from mon.core.size import Size

logger = get_logger(__name__)

type Target = dict[str, Any]
type SizeLike = Size | tuple[int, int]


# --- Helpers ---

def _size(canvas_size: SizeLike) -> Size:
    return canvas_size if isinstance(canvas_size, Size) else Size.from_any(canvas_size)


def _boxes(xyxy: list[list[float]] | torch.Tensor, size: Size) -> tv_tensors.BoundingBoxes:
    data = torch.as_tensor(xyxy, dtype=torch.float32).reshape(-1, 4)
    return tv_tensors.BoundingBoxes(data, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=size.hw)


def _empty_target(size: Size, *extra: str) -> Target:
    target: Target = {"boxes": _boxes([], size), "labels": torch.zeros(0, dtype=torch.int64)}
    for key in extra:
        target[key] = torch.zeros(0, dtype=torch.float32 if key == "scores" else torch.bool)
    return target


def _class_id(token: str, where: str) -> int:
    try:
        value = float(token)
    except ValueError:
        raise ValueError(f"{where}: class id {token!r} is not a number") from None
    if not value.is_integer() or value < 0:
        raise ValueError(f"{where}: class id must be a non-negative integer, got {token!r}")
    return int(value)


# --- YOLO ---

def read_yolo(path: str | os.PathLike[str], canvas_size: SizeLike, *, missing_ok: bool = True) -> Target:
    """Read a YOLO detection label file (``class cx cy w h [score]`` per line, normalized).

    Args:
        path: ``.txt`` label file.
        canvas_size: ``(H, W)`` of the image, to convert normalized coordinates to pixels.
        missing_ok: Treat a missing file as an image with no objects.

    Returns:
        ``{"boxes", "labels"}``, plus ``"scores"`` (float32) when any line has a 6th column.

    Raises:
        FileNotFoundError: If the file is missing and ``missing_ok`` is ``False``.
        ValueError: On a malformed line (wrong column count, non-numeric or non-finite values,
            bad class id, non-positive width/height, or a mix of 5- and 6-column lines).
    """
    size = _size(canvas_size)
    p = Path(path).expanduser()
    if not p.is_file():
        if missing_ok:
            return _empty_target(size)
        raise FileNotFoundError(f"label file not found: {p}")

    rows: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []
    ncols: int | None = None
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        where = f"{p}:{lineno}"
        if len(tokens) not in (5, 6):
            raise ValueError(f"{where}: expected 5 or 6 columns (class cx cy w h [score]), got {len(tokens)}")
        if ncols is None:
            ncols = len(tokens)
        elif len(tokens) != ncols:
            raise ValueError(f"{where}: mixes {ncols}- and {len(tokens)}-column lines")
        labels.append(_class_id(tokens[0], where))
        try:
            values = [float(t) for t in tokens[1:]]
        except ValueError:
            raise ValueError(f"{where}: non-numeric coordinate in {line!r}") from None
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{where}: non-finite value in {line!r}")
        cx, cy, w, h = values[:4]
        if w <= 0 or h <= 0:
            raise ValueError(f"{where}: width and height must be positive, got w={w}, h={h}")
        rows.append([(cx - w / 2) * size.w, (cy - h / 2) * size.h, (cx + w / 2) * size.w, (cy + h / 2) * size.h])
        if len(values) == 5:
            scores.append(values[4])

    if rows:
        target = {"boxes": _boxes(rows, size), "labels": torch.tensor(labels, dtype=torch.int64)}
    else:
        target = _empty_target(size)
    if ncols == 6:
        target["scores"] = torch.tensor(scores, dtype=torch.float32)
    return target


def write_yolo(
    path     : str | os.PathLike[str],
    boxes    : tv_tensors.BoundingBoxes,
    labels   : torch.Tensor | Sequence[int],
    scores   : torch.Tensor | Sequence[float] | None = None,
    *,
    precision: int = 6,
    overwrite: bool = True,
) -> Path:
    """Atomically write boxes as a YOLO label file (normalized ``class cx cy w h [score]``).

    Args:
        path: Destination ``.txt``; parent dirs are created.
        boxes: Boxes in any format; ``canvas_size`` gives the normalization.
        labels: One class id per box.
        scores: Optional confidence per box (6th column).
        precision: Decimal places for coordinates and scores.
        overwrite: Replace an existing file.

    Raises:
        ValueError: If lengths differ or ``boxes`` isn't a ``BoundingBoxes``.
    """
    if not isinstance(boxes, tv_tensors.BoundingBoxes):
        raise ValueError("boxes must be a tv_tensors.BoundingBoxes (it carries the canvas size)")
    from torchvision.transforms.v2.functional import convert_bounding_box_format

    h, w = boxes.canvas_size
    cxcywh = convert_bounding_box_format(boxes, new_format=tv_tensors.BoundingBoxFormat.CXCYWH)
    cxcywh = cxcywh.as_subclass(torch.Tensor)
    cxcywh = cxcywh.reshape(-1, 4).to(torch.float64) / torch.tensor([w, h, w, h], dtype=torch.float64)
    ids = torch.as_tensor(labels).reshape(-1).tolist()
    confs = None if scores is None else torch.as_tensor(scores).reshape(-1).tolist()
    if len(ids) != len(cxcywh) or (confs is not None and len(confs) != len(cxcywh)):
        raise ValueError(f"got {len(cxcywh)} boxes, {len(ids)} labels, {None if confs is None else len(confs)} scores")

    lines = []
    for i, (cx, cy, bw, bh) in enumerate(cxcywh.tolist()):
        cols = [str(int(ids[i]))] + [f"{v:.{precision}f}" for v in (cx, cy, bw, bh)]
        if confs is not None:
            cols.append(f"{confs[i]:.{precision}f}")
        lines.append(" ".join(cols) + "\n")
    target = _prepare_target(path, overwrite)
    _atomic_write_text(target, "".join(lines))
    return target


# --- Pascal VOC ---

def read_voc(
    path       : str | os.PathLike[str],
    classes    : Mapping[str, int] | Sequence[str],
    *,
    canvas_size: SizeLike | None = None,
    missing_ok : bool = False,
) -> Target:
    """Read a Pascal VOC XML annotation.

    VOC pixel indices are 1-based and inclusive; they are converted to 0-based continuous
    coordinates (``x1 = xmin - 1``, ``x2 = xmax``), as Detectron2 does.

    Args:
        path: ``.xml`` file.
        classes: Class name → id, or a sequence whose index is the id.
        canvas_size: ``(H, W)``; defaults to the file's ``<size>``. Required if that is missing.
        missing_ok: Treat a missing file as an image with no objects (needs ``canvas_size``).

    Returns:
        ``{"boxes", "labels", "difficult"}`` (``difficult`` is a bool tensor).

    Raises:
        FileNotFoundError: If the file is missing and ``missing_ok`` is ``False``.
        ValueError: On XML errors, unknown class names, missing fields, or bad coordinates.
    """
    class_ids = dict(classes) if isinstance(classes, Mapping) else {name: i for i, name in enumerate(classes)}
    p = Path(path).expanduser()
    if not p.is_file():
        if missing_ok and canvas_size is not None:
            return _empty_target(_size(canvas_size), "difficult")
        if missing_ok:
            raise ValueError(f"{p} is missing; missing_ok=True needs canvas_size")
        raise FileNotFoundError(f"annotation file not found: {p}")
    try:
        root = ET.parse(p).getroot()
    except ET.ParseError as err:
        raise ValueError(f"{p}: invalid XML: {err}") from None

    def text(node: ET.Element, tag: str, where: str) -> str:
        child = node.find(tag)
        if child is None or child.text is None:
            raise ValueError(f"{where}: missing <{tag}>")
        return child.text.strip()

    if canvas_size is None:
        size_node = root.find("size")
        if size_node is None:
            raise ValueError(f"{p}: no <size>; pass canvas_size")
        size = Size(int(text(size_node, "height", str(p))), int(text(size_node, "width", str(p))))
    else:
        size = _size(canvas_size)

    rows: list[list[float]] = []
    labels: list[int] = []
    difficult: list[bool] = []
    for i, obj in enumerate(root.findall("object")):
        where = f"{p}: object {i}"
        name = text(obj, "name", where)
        if name not in class_ids:
            raise ValueError(f"{where}: unknown class {name!r}; known: {sorted(class_ids)}")
        bnd = obj.find("bndbox")
        if bnd is None:
            raise ValueError(f"{where}: missing <bndbox>")
        try:
            xmin, ymin, xmax, ymax = (float(text(bnd, t, where)) for t in ("xmin", "ymin", "xmax", "ymax"))
        except ValueError as err:
            raise ValueError(f"{where}: bad coordinate: {err}") from None
        if xmin < 1 or ymin < 1 or xmax < xmin or ymax < ymin:
            raise ValueError(f"{where}: invalid box xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
        rows.append([xmin - 1, ymin - 1, xmax, ymax])
        labels.append(class_ids[name])
        diff = obj.find("difficult")
        difficult.append(diff is not None and (diff.text or "0").strip() == "1")

    if not rows:
        return _empty_target(size, "difficult")
    return {
        "boxes"    : _boxes(rows, size),
        "labels"   : torch.tensor(labels, dtype=torch.int64),
        "difficult": torch.tensor(difficult, dtype=torch.bool),
    }


# --- COCO ---

@dataclass(frozen=True)
class CocoAnnotations:
    """Box annotations from a COCO JSON file, indexed by image ``file_name``.

    Category ids are kept as in the file (COCO's are 1–90 with gaps); map them to contiguous
    training ids in the dataset if needed. ``num_skipped`` counts annotations dropped for a
    non-positive width or height (official COCO files contain a few).
    """

    categories : Mapping[int, str]
    images     : Mapping[str, Size]
    _anns      : Mapping[str, list[dict[str, Any]]] = field(repr=False)
    num_skipped: int = 0

    def __len__(self) -> int:
        return len(self.images)

    def __contains__(self, file_name: object) -> bool:
        return file_name in self.images

    def target(self, file_name: str) -> Target:
        """Return ``{"boxes", "labels", "iscrowd"}`` for an image (empty if it has no annotations).

        Raises:
            KeyError: If ``file_name`` isn't one of the file's images.
        """
        if file_name not in self.images:
            raise KeyError(f"{file_name!r} is not in the COCO file")
        size = self.images[file_name]
        anns = self._anns.get(file_name, [])
        if not anns:
            return _empty_target(size, "iscrowd")
        rows = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        return {
            "boxes"  : _boxes(rows, size),
            "labels" : torch.tensor([a["category_id"] for a in anns], dtype=torch.int64),
            "iscrowd": torch.tensor([bool(a.get("iscrowd", 0)) for a in anns], dtype=torch.bool),
        }


def read_coco(path: str | os.PathLike[str]) -> CocoAnnotations:
    """Read a COCO detection JSON (``images``, ``annotations``, ``categories``); boxes only.

    Annotations with a non-positive width or height are skipped and counted
    (``CocoAnnotations.num_skipped``, plus one warning); everything else malformed raises.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: On missing keys, duplicate image ids/names, or annotations pointing at
            unknown images or categories.
    """
    data = load_json(path)
    where = str(Path(path).expanduser())
    try:
        images, anns, cats = data["images"], data.get("annotations", []), data["categories"]
    except (KeyError, TypeError):
        raise ValueError(f"{where}: expected a COCO dict with 'images' and 'categories'") from None

    categories = {int(c["id"]): str(c["name"]) for c in cats}
    id_to_name: dict[int, str] = {}
    sizes: dict[str, Size] = {}
    for img in images:
        image_id, name = int(img["id"]), str(img["file_name"])
        if image_id in id_to_name or name in sizes:
            raise ValueError(f"{where}: duplicate image id {image_id} or file_name {name!r}")
        id_to_name[image_id] = name
        sizes[name] = Size(int(img["height"]), int(img["width"]))

    per_image: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for ann in anns:
        aid = ann.get("id")
        if ann["image_id"] not in id_to_name:
            raise ValueError(f"{where}: annotation {aid} refers to unknown image_id {ann['image_id']}")
        if ann["category_id"] not in categories:
            raise ValueError(f"{where}: annotation {aid} has unknown category_id {ann['category_id']}")
        x, y, w, h = (float(v) for v in ann["bbox"])
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        per_image.setdefault(id_to_name[ann["image_id"]], []).append({**ann, "bbox": (x, y, w, h)})
    if skipped:
        logger.warning("%s: skipped %d annotation(s) with non-positive width or height", where, skipped)
    return CocoAnnotations(categories=categories, images=sizes, _anns=per_image, num_skipped=skipped)
