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

Inputs and targets of a pair get the identical resize and crop, so they stay aligned. Stems are
``<source dataset>_<source stem>`` (``/`` → ``_``), e.g. ``lol_v1_111``, ``sice_me_12_3``.
``build`` writes ``manifest.yaml`` (source path and SHA-256 of every source and output file), so any
machine rebuilds the same files: ``python -m mon.data.zoo testkit build``. New tasks add a dataset here
when their data and models exist.

Use it for regression tests (seconds, any device); the full datasets stay for parity with papers.
Scores on it are not comparable to published numbers.

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> sorted(n for n in DATASETS.names() if n.startswith("testkit"))
    ['testkit_classify', 'testkit_lle', 'testkit_lle_unpaired', 'testkit_mef']
    >>> TestkitMEF.pair_key("target", "sice_3/1")
    'sice_3'
"""

__all__ = [
    "SIZE",
    "TestkitClassify",
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


def build(out: str | Path | None = None, *, overwrite: bool = False) -> Path:
    """Build the testkit from the source datasets (all must be on this machine); return its root.

    Raises:
        FileExistsError: If ``out`` exists and ``overwrite`` is false (nothing is ever deleted).
        FileNotFoundError / ValueError: Missing source datasets or files.
    """
    root = Path(out) if out is not None else datasets_dir() / "testkit"
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"{root} exists; pass overwrite=True (files are replaced, none deleted)")
    w = _Writer(root)
    for part in (_lle, _lle_unpaired, _mef, _classify):
        part(w)
        logger.info("testkit: %s done (%d files so far)", part.__name__.strip("_"), len(w.entries))
    save_yaml({"size": SIZE, "resize": "short side to size (cv2.INTER_AREA), centre crop; classify: native",
               "files": w.entries}, root / "manifest.yaml")
    return root
