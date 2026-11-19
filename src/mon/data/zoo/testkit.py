"""Testkit: a small, fixed subset of every task's data, for fast tests (not a benchmark).

100 samples per dataset, drawn deterministically from the registered datasets, resized so the short side
is 256 (OpenCV ``INTER_AREA``) and centre-cropped to 256×256 (classification keeps its native
size), stored as PNG under ``datasets_dir() / "testkit"``. One dataset id per task family:

- ``testkit_lle``: paired low/normal light, 100 pairs: LOL-v1 test (all 15), LOL-v2-real test (40, evenly
  spaced), SICE-ME val (45, evenly spaced). No training images: the models were trained on them.
- ``testkit_lle_unpaired``: no-reference low light, 100 images: all of Fusion (16), LIME (10), MEF (17), NPE (8),
  VV (24), and 25 of DICM (evenly spaced).
- ``testkit_mef``: exposure stacks, 25 SICE test scenes (evenly spaced among those with ≥ 4 exposures) × 4
  exposures (darkest, two in between, brightest) = 100, plus each scene's reference.
- ``testkit_classify``: CIFAR-10 test, the first 10 images of each class = 100 (32×32, native).
- ``testkit_efficiency``: model efficiency (params, MACs, speed, memory) across resolutions, not quality: the same
  5 dark SICE test scenes (darkest exposure; scenes with a landscape photo of at least 3840×2160, evenly spaced) at
  each of ``EFFICIENCY_SIZES`` (256×256 to 8K), paired with the scene's reference: ``test/images/<W>x<H>/sice_<scene>.png``
  and ``test/targets/<W>x<H>/sice_<scene>.png``. Both get the identical scaling to cover the size and centre crop:
  ``INTER_AREA`` down, bicubic up (only 8K, which exceeds every SICE photo).
  ``mon benchmark data=testkit_efficiency`` measures a model on it.

Inputs and targets of a pair get the identical resize and crop, so they stay aligned. Stems are
``<source dataset>_<source stem>`` (``/`` → ``_``), e.g. ``lol_v1_111``, ``sice_me_12_3``.
``build`` writes ``manifest.yaml`` (source path and SHA-256 of every source and output file), so any
machine rebuilds the same files: ``python -m mon.data.zoo testkit build`` (``--part efficiency`` rebuilds one
part and updates its manifest entries). New tasks add a dataset here
when their data and models exist.

Use it for regression tests (seconds, any device); the full datasets stay for parity with papers.
Scores on it are not comparable to published numbers.

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> sorted(n for n in DATASETS.names() if n.startswith("testkit"))
    ['testkit_classify', 'testkit_efficiency', 'testkit_lle', 'testkit_lle_unpaired', 'testkit_mef']
    >>> TestkitMEF.pair_key("target", "sice_3/1")
    'sice_3'
"""

__all__ = [
    "EFFICIENCY_SIZES",
    "PARTS",
    "SIZE",
    "TestkitClassify",
    "TestkitEfficiency",
    "TestkitLLE",
    "TestkitLLEUnpaired",
    "TestkitMEF",
    "build",
]

import hashlib
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from mon.core.console import get_logger
from mon.core.enums import Task
from mon.core.fileio import save_yaml
from mon.core.path import Path, datasets_dir
from mon.data.dataset import (
    ClassFolderDataset, FolderDataset, classification_layout, paired_layout, register_dataset, unpaired_layout,
)

logger = get_logger(__name__)

SIZE = 256
"""Side of the square testkit images (classification keeps its native size)."""

_LLE_PAIRED    = (("lol_v1", "test", 15), ("lol_v2_real", "test", 40), ("sice_me", "val", 45))
_UNPAIRED_SETS = (("dicm", 25), ("fusion", 16), ("lime", 10), ("mef", 17), ("npe", 8), ("vv", 24))
_MEF_SCENES, _MEF_EXPOSURES, _PER_CLASS = 25, 4, 10
_EFFICIENCY_SCENES = 5

EFFICIENCY_SIZES: tuple[tuple[int, int], ...] = (
    (256, 256), (512, 512), (600, 400), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160), (7680, 4320),
)
"""``(W, H)`` of ``testkit_efficiency``: square sizes papers use, LOL-v1's native size, 720p, 1080p, 2K, 4K, 8K."""


# --- Datasets ---

@register_dataset("testkit_lle", tasks={Task.LLE})
class TestkitLLE(FolderDataset):
    """Testkit paired low-light enhancement: 100 pairs (LOL-v1, LOL-v2-real, SICE-ME), 256×256."""

    __test__ = False  # "Test…" names: keep pytest from collecting these classes
    folder = "testkit/lle"
    layout = paired_layout("test")


@register_dataset("testkit_lle_unpaired", tasks={Task.LLE})
class TestkitLLEUnpaired(FolderDataset):
    """Testkit no-reference low light: 100 images from the six unpaired LOL sets, 256×256."""

    __test__ = False  # "Test…" names: keep pytest from collecting these classes
    folder = "testkit/lle_unpaired"
    layout = unpaired_layout("test")


@register_dataset("testkit_mef", tasks={Task.MEF})
class TestkitMEF(FolderDataset):
    """Testkit exposure stacks: 25 SICE test scenes × 4 exposures and their references, 256×256.

    Stored like ``sice`` (``test/images/<scene>/<k>.png``, ``test/targets/<scene>.png``); one sample
    per exposure, ``meta.stem`` = ``"<scene>/<k>"`` (stack loading for MEF models comes later).
    """

    __test__ = False  # "Test…" names: keep pytest from collecting these classes
    folder = "testkit/mef"
    layout = paired_layout("test")

    @classmethod
    def pair_key(cls, modality: str, key: str) -> str:
        """Exposure ``"<scene>/<k>"`` → target ``"<scene>"``."""
        return key.split("/", 1)[0] if modality == "target" else key


@register_dataset("testkit_classify", tasks={Task.CLASSIFY})
class TestkitClassify(ClassFolderDataset):
    """Testkit classification: CIFAR-10 test, the first 10 images of each class (32×32)."""

    __test__ = False  # "Test…" names: keep pytest from collecting these classes
    folder = "testkit/classify"
    layout = classification_layout("test")


@register_dataset("testkit_efficiency", tasks={Task.LLE})
class TestkitEfficiency(FolderDataset):
    """Testkit efficiency ladder: 5 dark SICE scenes and their references at each of ``EFFICIENCY_SIZES``.

    ``meta.stem`` = ``"<W>x<H>/<scene>"``.
    """

    __test__ = False  # "Test…" names: keep pytest from collecting these classes
    folder = "testkit/efficiency"
    layout = paired_layout("test")


# --- Builder ---

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot decode {path}")
    return img


def _square(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Resize the short side to ``size`` (area) and centre-crop to ``size × size``."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, round(h * scale)), max(size, round(w * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top, left = (nh - size) // 2, (nw - size) // 2
    return img[top:top + size, left:left + size]


def _cover(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale to cover ``width × height`` (area down, bicubic up) and centre-crop to it."""
    h, w = img.shape[:2]
    scale = max(width / w, height / h)
    nw, nh = max(width, round(w * scale)), max(height, round(h * scale))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    top, left = (nh - height) // 2, (nw - width) // 2
    return img[top:top + height, left:left + width]


def _evenly(items: Sequence[Any], count: int) -> list[Any]:
    """``count`` items evenly spaced through ``items`` (all of them if there are fewer)."""
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)] if count > 1 else [items[0]]


class _Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.data = datasets_dir()
        self.entries: list[dict[str, str]] = []

    def write(self, img: np.ndarray, rel: str, source: Path) -> None:
        out = self.root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), img):
            raise OSError(f"cannot write {out}")
        self.entries.append({"file": rel, "source": str(source.relative_to(self.data)),
                             "source_sha256": _sha256(source), "sha256": _sha256(out)})


def _stem(name: str, stem: str) -> str:
    return f"{name}_{stem.replace('/', '_')}"


def _lle(w: _Writer) -> None:
    from mon.data.dataset import DATASETS

    for name, split, count in _LLE_PAIRED:
        items = _evenly(sorted(DATASETS.build(name, split).items(), key=lambda kv: kv[0]), count)
        for stem, files in items:
            key = _stem(name, stem)
            w.write(_square(_read(files["image"])), f"lle/test/images/{key}.png", files["image"])
            w.write(_square(_read(files["target"])), f"lle/test/targets/{key}.png", files["target"])


def _lle_unpaired(w: _Writer) -> None:
    from mon.data.dataset import DATASETS

    for name, count in _UNPAIRED_SETS:
        for stem, files in _evenly(sorted(DATASETS.build(name, "test").items(), key=lambda kv: kv[0]), count):
            w.write(_square(_read(files["image"])), f"lle_unpaired/test/images/{_stem(name, stem)}.png", files["image"])


def _mef(w: _Writer) -> None:
    from mon.data.dataset import DATASETS

    scenes: dict[str, list[tuple[str, Any]]] = {}
    for stem, files in DATASETS.build("sice", "test").items():
        scene, k = stem.split("/", 1)
        scenes.setdefault(scene, []).append((k, files))
    eligible = sorted((s for s in scenes if len(scenes[s]) >= _MEF_EXPOSURES), key=lambda s: (len(s), s))  # numeric
    for scene in _evenly(eligible, _MEF_SCENES):
        exposures = sorted(scenes[scene], key=lambda kv: (len(kv[0]), kv[0]))  # k = 1 (darkest) … n (brightest)
        picked = _evenly(exposures, _MEF_EXPOSURES)
        for i, (k, files) in enumerate(picked, start=1):
            w.write(_square(_read(files["image"])), f"mef/test/images/sice_{scene}/{i}.png", files["image"])
        target = picked[0][1]["target"]
        w.write(_square(_read(target)), f"mef/test/targets/sice_{scene}.png", target)


def _efficiency(w: _Writer) -> None:
    from mon.data.dataset import DATASETS

    darkest: dict[str, Path] = {}
    targets: dict[str, Path] = {}
    for stem, files in DATASETS.build("sice", "test").items():
        scene, k = stem.split("/", 1)
        if k == "1":
            darkest[scene], targets[scene] = files["image"], files["target"]
    scenes = sorted(darkest, key=lambda s: (len(s), s))                                  # numeric order
    min_w, min_h = EFFICIENCY_SIZES[-2]                                                   # 4K: largest pure downscale
    picked: list[tuple[str, np.ndarray]] = []
    for start in (round(i * len(scenes) / _EFFICIENCY_SCENES) for i in range(_EFFICIENCY_SCENES)):
        for scene in scenes[start:]:                                                      # first fit from each start
            if scene in {p for p, _ in picked}:
                continue
            img = _read(darkest[scene])
            if img.shape[1] >= max(img.shape[0], min_w) and img.shape[0] >= min_h:
                picked.append((scene, img))
                break
    if len(picked) < _EFFICIENCY_SCENES:
        raise ValueError(f"found only {len(picked)} SICE test scenes with a landscape photo >= {min_w}×{min_h}")
    refs = {scene: _read(targets[scene]) for scene, _ in picked}
    for scene, img in picked:
        if refs[scene].shape != img.shape:
            raise ValueError(f"SICE test scene {scene}: target {refs[scene].shape} differs from input {img.shape}")
    for width, height in EFFICIENCY_SIZES:
        for scene, img in picked:
            rel = f"{width}x{height}/sice_{scene}.png"
            w.write(_cover(img, width, height), f"efficiency/test/images/{rel}", darkest[scene])
            w.write(_cover(refs[scene], width, height), f"efficiency/test/targets/{rel}", targets[scene])


def _classify(w: _Writer) -> None:
    from mon.data.dataset import DATASETS

    ds = DATASETS.build("cifar10", "test")
    names = list(ds.classes.names)
    taken: dict[str, int] = dict.fromkeys(names, 0)
    for stem, files in ds.items():  # stems "<class>/<index>", in dataset order
        cls = stem.split("/", 1)[0]
        if taken[cls] < _PER_CLASS:
            taken[cls] += 1
            w.write(_read(files["image"]), f"classify/test/{cls}/{_stem('cifar10', stem.split('/', 1)[1])}.png",
                    files["image"])
    classes = w.root / "classify" / "classes.txt"
    classes.write_text("\n".join(names) + "\n")


PARTS: dict[str, str] = {"lle": "lle", "lle_unpaired": "lle_unpaired", "mef": "mef", "classify": "classify",
                         "efficiency": "efficiency"}
"""Builder parts → their folder under the testkit root."""

_BUILDERS = {"lle": _lle, "lle_unpaired": _lle_unpaired, "mef": _mef, "classify": _classify, "efficiency": _efficiency}
_RESIZE = ("short side to size (cv2.INTER_AREA), centre crop; classify: native; efficiency: cover each size "
           "(INTER_AREA down, INTER_CUBIC up), centre crop")


def build(out: str | Path | None = None, *, overwrite: bool = False, parts: Sequence[str] | None = None) -> Path:
    """Build the testkit from the source datasets (all must be on this machine); return its root.

    ``parts`` (keys of ``PARTS``; default all) rebuilds only those: their manifest entries are replaced, the
    others' kept.

    Raises:
        FileExistsError: If a part's folder (or, for a full build, ``out``) has files and ``overwrite`` is false
            (nothing is ever deleted).
        FileNotFoundError / ValueError: Missing source datasets or files, or an unknown part.
    """
    root = Path(out) if out is not None else datasets_dir() / "testkit"
    names = list(PARTS) if parts is None else list(parts)
    unknown = sorted(set(names) - set(PARTS))
    if unknown:
        raise ValueError(f"unknown testkit parts {unknown}; choose from {list(PARTS)}")
    targets = [root] if parts is None else [root / PARTS[n] for n in names]
    for target in targets:
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise FileExistsError(f"{target} exists; pass overwrite=True (files are replaced, none deleted)")
    manifest = root / "manifest.yaml"
    kept: list[dict[str, str]] = []
    if parts is not None and manifest.is_file():
        from mon.core.fileio import load_yaml

        prefixes = tuple(f"{PARTS[n]}/" for n in names)
        kept = [e for e in load_yaml(manifest)["files"] if not e["file"].startswith(prefixes)]
    w = _Writer(root)
    for name in names:
        _BUILDERS[name](w)
        logger.info("testkit: %s done (%d files so far)", name, len(w.entries))
    entries = sorted(kept + w.entries, key=lambda e: [*PARTS.values()].index(e["file"].split("/", 1)[0]))
    save_yaml({"size": SIZE, "efficiency_sizes": [f"{a}x{b}" for a, b in EFFICIENCY_SIZES], "resize": _RESIZE,
               "files": entries}, manifest)
    return root
