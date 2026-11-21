"""Dataset overview images: ``<split>/overview.jpg`` with random samples and their annotations, for a visual check.

Usage (from a mon checkout)::

    uv run python tools/dataset_overview.py lol_v1                 # every split of a registered dataset
    uv run python tools/dataset_overview.py fisheye8k --split val --n 30 --seed 1
    uv run python tools/dataset_overview.py path/to/folder         # any image folder (images only)

Each split gets ``<dataset root>/<split>/overview.jpg`` (``<folder>/overview.jpg`` for a folder): ``n`` random
items (``seed``), a title naming what is drawn, and a caption per tile. An existing overview is kept unless
``--overwrite``. Views, picked from the dataset's layout and samples:

- **sample** (default): the input, with ``boxes`` (+ ``labels``, a class legend), the ``label`` class name, any
  ``Mask`` (colorized) and other image modalities (``target``, …) side by side; caption ``<stem>``.
- **sequence**: inputs stored as ``images/<group>/<k>`` with one target per group (``targets/<group>``), e.g. SICE
  exposures or LoLI-Street levels: one strip per group, the inputs (``--max-seq`` evenly spaced) then the target.
- **mot_test**: MOTChallenge test splits (images only in mon): frames from ``seq/<seq>/img1/`` with the public
  detections from ``seq/<seq>/det/det.txt``.

Not data: the file sits directly in the split folder, where no mon loader reads samples.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torchvision import tv_tensors

from mon.data.dataset import DATASETS, FolderDataset, ImageFolder
from mon.data.io import read_image

Image = NDArray[np.uint8]  # H×W×3, BGR (OpenCV)

BG, FG   = (32, 32, 32), (235, 235, 235)
PALETTE  = [(0, 255, 0), (64, 64, 255), (255, 160, 64), (0, 200, 255), (255, 0, 255), (255, 255, 0), (0, 128, 255)]
FONT     = cv2.FONT_HERSHEY_SIMPLEX
MAX_W    = 2600
TILE_H   = 300

__all__ = ["overview", "write_overviews"]


# --- Drawing ---

def to_bgr(image: torch.Tensor) -> Image:
    """``[C, H, W]`` image tensor (uint8, or float in ``[0, 1]``; 1 or 3 channels) → BGR uint8 array."""
    t = image.detach().cpu()
    if t.is_floating_point():
        t = (t.clamp(0, 1) * 255).round()
    a = t.to(torch.uint8).numpy()
    a = np.repeat(a, 3, axis=0) if a.shape[0] == 1 else a[:3]
    return np.ascontiguousarray(a.transpose(1, 2, 0)[:, :, ::-1])


def fit(image: Image, height: int) -> Image:
    """Resize to ``height`` keeping the aspect ratio (area interpolation when shrinking)."""
    h, w = image.shape[:2]
    width = max(1, round(w * height / h))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_NEAREST
    return np.asarray(cv2.resize(image, (width, height), interpolation=interp), dtype=np.uint8)


def hstack(images: Sequence[Image], gap: int = 4) -> Image:
    h = max(i.shape[0] for i in images)
    out = np.full((h, sum(i.shape[1] for i in images) + gap * (len(images) - 1), 3), BG, np.uint8)
    x = 0
    for i in images:
        out[: i.shape[0], x : x + i.shape[1]] = i
        x += i.shape[1] + gap
    return out


def draw_boxes(image: Image, boxes: Iterable[Sequence[float]], labels: Sequence[int], scale: float) -> Image:
    """Draw ``XYXY`` pixel boxes (in the original image's pixels) on ``image``, resized by ``scale``."""
    out = image.copy()
    thick = max(1, image.shape[0] // 150)
    for (x0, y0, x1, y1), c in zip(boxes, labels, strict=True):
        p0 = (round(x0 * scale), round(y0 * scale))
        p1 = (round(x1 * scale), round(y1 * scale))
        cv2.rectangle(out, p0, p1, PALETTE[c % len(PALETTE)], thick)
    return out


def colorize(mask: torch.Tensor) -> Image:
    """Class-id mask → BGR (a fixed color per id; 255, the usual "ignore", is black)."""
    m = mask.detach().cpu().numpy().astype(np.int64)
    rng = np.random.default_rng(0)
    lut = rng.integers(40, 256, size=(256, 3), dtype=np.uint8)
    lut[255] = 0
    return np.ascontiguousarray(lut[np.clip(m, 0, 255)])


def text(img: Image, s: str, org: tuple[int, int], scale: float, center: bool = False) -> None:
    (w, _), _ = cv2.getTextSize(s, FONT, scale, 1)
    x = org[0] - w // 2 if center else org[0]
    cv2.putText(img, s, (x, org[1]), FONT, scale, FG, 1 if scale < 1 else 2, cv2.LINE_AA)


def shorten(s: str, width: int, scale: float) -> str:
    """Cut ``s`` in the middle (``…`` → ``..``) until it fits ``width`` pixels."""
    if cv2.getTextSize(s, FONT, scale, 1)[0][0] <= width:
        return s
    k = len(s)
    while k > 4 and cv2.getTextSize(s[: k // 2] + ".." + s[-(k // 2) :], FONT, scale, 1)[0][0] > width:
        k -= 2
    return s[: k // 2] + ".." + s[-(k // 2) :]


def compose(tiles: Sequence[tuple[Image, str]], title: str, legend: Sequence[str] = ()) -> Image:
    """Grid of ``(tile, caption)`` under ``title`` (and a box-color legend), at most ``MAX_W`` wide."""
    tw = max(t.shape[1] for t, _ in tiles)
    th = max(t.shape[0] for t, _ in tiles)
    cols = max(1, min(len(tiles), MAX_W // (tw + 10)))
    rows = (len(tiles) + cols - 1) // cols
    top, cap_h = 64 + (36 if legend else 0), 32
    out = np.full((top + rows * (th + cap_h + 10), cols * (tw + 10) + 10, 3), BG, np.uint8)
    text(out, title, (out.shape[1] // 2, 44), 1.1, center=True)
    x = 20
    for k, name in enumerate(legend):
        cv2.rectangle(out, (x, 72), (x + 22, 94), PALETTE[k % len(PALETTE)], 3)
        text(out, name, (x + 30, 91), 0.6)
        x += 50 + cv2.getTextSize(name, FONT, 0.6, 1)[0][0]
    for k, (t, cap) in enumerate(tiles):
        r, c = divmod(k, cols)
        x0, y0 = 10 + c * (tw + 10), top + r * (th + cap_h + 10)
        dx = (tw - t.shape[1]) // 2
        out[y0 : y0 + t.shape[0], x0 + dx : x0 + dx + t.shape[1]] = t
        text(out, shorten(cap, tw, 0.6), (x0 + tw // 2, y0 + th + 24), 0.6, center=True)
    if out.shape[1] > MAX_W + 20:
        out = fit(out, round(out.shape[0] * MAX_W / out.shape[1]))
    return out


# --- Views ---

def natural(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def pick(keys: Sequence[str], n: int, seed: int) -> list[str]:
    """``n`` random keys (all when fewer), in natural order."""
    return sorted(random.Random(seed).sample(list(keys), min(n, len(keys))), key=natural)


def sample_view(ds: Any, n: int, seed: int, name: str) -> Image:
    stems = [stem for stem, _ in ds.items()]
    index = {stem: i for i, stem in enumerate(stems)}
    stems = [s for s in stems if s.rsplit("/", 1)[-1] != "overview"]  # an earlier overview is not a sample
    classes = list(ds.classes.names) if getattr(ds, "classes", None) is not None else []
    tiles: list[tuple[Image, str]] = []
    drawn: set[str] = set()
    for stem in pick(stems, n, seed):
        s = ds[index[stem]]
        img = to_bgr(s["image"])
        scale = TILE_H / img.shape[0]
        panel, cap = fit(img, TILE_H), stem
        if "boxes" in s:
            labels = [int(v) for v in s.get("labels", torch.zeros(len(s["boxes"]), dtype=torch.int64))]
            panel = draw_boxes(panel, s["boxes"].tolist(), labels, scale)
            cap, drawn = f"{stem} ({len(labels)})", drawn | {"boxes"}
        if "label" in s:
            c = int(s["label"])
            cap, drawn = (classes[c] if c < len(classes) else str(c)), drawn | {"class"}
        panels = [panel]
        for key, v in s.items():
            if key in ("image", "boxes", "labels", "label", "meta") or not isinstance(v, torch.Tensor):
                continue
            if isinstance(v, tv_tensors.Mask):
                panels.append(fit(colorize(v), TILE_H))
            elif isinstance(v, tv_tensors.Image):
                panels.append(fit(to_bgr(v), TILE_H))
            else:
                continue
            drawn.add(key)
        tiles.append((hstack(panels) if len(panels) > 1 else panels[0], cap))
    others = [k for k in drawn if k not in ("boxes", "class")]
    what = ("boxes" if "boxes" in drawn else "class" if "class" in drawn else "")
    parts = (["input | " + " | ".join(sorted(others))] if others else []) + ([what] if what else [])
    return compose(tiles, f"{name}: {len(tiles)} random items (seed {seed}), {', '.join(parts) or 'images'}",
                   classes if "boxes" in drawn else ())


def is_sequence(ds: Any) -> bool:
    """Inputs stored as ``<group>/<k>`` whose target key is ``<group>`` (multi-exposure / multi-level layout)."""
    if not isinstance(ds, FolderDataset) or "target" not in ds.modalities:
        return False
    items = ds.items()
    return bool(items) and all("/" in stem and type(ds).pair_key("target", stem) == stem.rsplit("/", 1)[0]
                               for stem, _ in items)


def sequence_view(ds: Any, n: int, seed: int, name: str, max_seq: int) -> Image:
    groups: dict[str, list[tuple[str, Any]]] = {}
    for stem, paths in ds.items():
        groups.setdefault(stem.rsplit("/", 1)[0], []).append((stem, paths))
    tiles = []
    for g in pick(list(groups), n, seed):
        members = sorted(groups[g], key=lambda m: natural(m[0]))
        if len(members) > max_seq:
            members = [members[round(i * (len(members) - 1) / (max_seq - 1))] for i in range(max_seq)]
        h = 150
        strip = [fit(to_bgr(read_image(p["image"], "rgb")), h) for _, p in members]
        strip.append(fit(to_bgr(read_image(members[0][1]["target"], "rgb")), h))
        ks = ", ".join(m[0].rsplit("/", 1)[1] for m in members)
        tiles.append((hstack(strip, 3), f"{g}: {ks} | target"))
    return compose(tiles, f"{name}: {len(tiles)} random groups (seed {seed}), inputs | target "
                          f"(> {max_seq}: {max_seq} evenly spaced)")


def is_mot_test(ds: Any) -> bool:
    if not isinstance(ds, FolderDataset):
        return False
    seq = ds.root / next(iter(ds.modalities.values())).dir
    return seq.name == "seq" and any((d / "det" / "det.txt").is_file() for d in seq.iterdir() if d.is_dir())


def mot_test_view(ds: Any, n: int, seed: int, name: str) -> Image:
    seq_dir = ds.root / next(iter(ds.modalities.values())).dir
    frames = [(stem, paths["image"]) for stem, paths in ds.items() if "/img1/" in stem]
    tiles = []
    dets: dict[str, NDArray[np.float64]] = {}
    for stem, path in sorted(random.Random(seed).sample(frames, min(n, len(frames))), key=lambda f: natural(f[0])):
        seq, frame = stem.split("/")[0], int(stem.rsplit("/", 1)[1])
        if seq not in dets:
            dets[seq] = np.loadtxt(seq_dir / seq / "det" / "det.txt", delimiter=",", ndmin=2)
        d = dets[seq][dets[seq][:, 0] == frame]
        img = to_bgr(read_image(path, "rgb"))
        scale = TILE_H / img.shape[0]
        boxes = [(x, y, x + w, y + h) for x, y, w, h in d[:, 2:6]]
        tiles.append((draw_boxes(fit(img, TILE_H), boxes, [0] * len(boxes), scale), f"{seq}/{frame} ({len(boxes)})"))
    return compose(tiles, f"{name}: {len(tiles)} random frames (seed {seed}), public detections (seq/*/det/det.txt)",
                   ["detection"])


def overview(ds: Any, *, n: int = 20, seed: int = 0, name: str = "", max_seq: int = 9) -> Image:
    """Overview image (BGR uint8) of dataset ``ds``, picking the view from its layout and samples."""
    if is_mot_test(ds):
        return mot_test_view(ds, n, seed, name)
    if is_sequence(ds):
        return sequence_view(ds, n, seed, name, max_seq)
    return sample_view(ds, n, seed, name)


# --- CLI ---

def split_dir(ds: Any) -> Path:
    """The split's folder: the first path part shared by its modality folders (``train/images`` → ``train``);
    for a ``ClassFolderDataset``, its layout's split folder."""
    if not isinstance(ds, FolderDataset):
        return Path(ds.root) / str(type(ds).layout[ds.split])
    parts = [Path(m.dir).parts for m in ds.modalities.values()]
    common = [p[0] for p in zip(*parts, strict=False) if len(set(p)) == 1]
    first = common[0] if common else parts[0][0]
    return ds.root / first


def write_overviews(spec: str, splits: Sequence[str] = (), *, n: int = 20, seed: int = 0, out: str | None = None,
                    overwrite: bool = False, max_seq: int = 9, log: Callable[[str], None] = print) -> list[Path]:
    """Write the overview of each split of dataset id (or image folder) ``spec``; return the files written."""
    jobs: list[tuple[Any, Path, str]] = []
    if spec in DATASETS:
        cls = DATASETS.get(spec)
        for s in splits or [x.value for x in cls.splits()]:
            ds = DATASETS.build(spec, s)
            jobs.append((ds, split_dir(ds) / "overview.jpg", f"{spec} {s}"))
    else:
        folder = Path(spec).expanduser()
        if not folder.is_dir():
            raise SystemExit(f"{spec!r} is neither a dataset id nor a folder")
        jobs.append((ImageFolder(folder), folder / "overview.jpg", folder.name))
    written = []
    for ds, path, name in jobs:
        path = Path(out) if out and len(jobs) == 1 else path
        if path.exists() and not overwrite:
            log(f"skip (exists): {path}")
            continue
        img = overview(ds, n=n, seed=seed, name=name, max_seq=max_seq)
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            raise OSError(f"could not write {path}")
        log(f"wrote {path} ({img.shape[1]}x{img.shape[0]})")
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("spec", help="dataset id (mon.data DATASETS) or an image folder")
    p.add_argument("--split", action="append", default=[], help="split (repeatable); default: all")
    p.add_argument("--n", type=int, default=20, help="items per overview")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-seq", type=int, default=9, help="inputs per group in the sequence view")
    p.add_argument("--out", help="output file (one split only); default: <split>/overview.jpg")
    p.add_argument("--overwrite", action="store_true", help="replace an existing overview")
    a = p.parse_args(argv)
    write_overviews(a.spec, a.split, n=a.n, seed=a.seed, out=a.out, overwrite=a.overwrite, max_seq=a.max_seq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
