"""Tests for ``mon.data.zoo.detect``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.detect import DarkFace, ExDark, detection_layout

CLASSES = {"exdark": ExDark, "darkface": DarkFace}
SPLITS = {"exdark": {Split.TRAIN, Split.VAL, Split.TEST}, "darkface": {Split.TRAIN, Split.TEST}}
EXPECTED = {
    "exdark"  : {Split.TRAIN: (3000, 10381), Split.VAL: (1800, 6212), Split.TEST: (2563, 7117)},
    "darkface": {Split.TRAIN: (5000, 41966), Split.TEST: (1000, 8430)},
}


def _write(root: StdPath, split: str, labels: dict[str, str]) -> None:
    for stem, text in labels.items():
        write_image(torch.zeros(3, 40, 60, dtype=torch.uint8), root / split / "images" / f"{stem}.png")
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        (root / split / "labels" / f"{stem}.txt").write_text(text)


@pytest.mark.parametrize(("name", "cls"), CLASSES.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.DETECT, Task.LLE}
    assert DATASETS.entry(name).meta["splits"] == SPLITS[name]
    assert cls.folder == name


def test_detection_layout():
    layout = detection_layout("train", Split.TEST)
    assert list(layout) == [Split.TRAIN, Split.TEST]
    assert [(m.dir, m.kind) for m in layout[Split.TEST].values()] == [("test/images", "image"), ("test/labels", "yolo")]
    with pytest.raises(ValueError, match="at least one"):
        detection_layout()
    with pytest.raises(ValueError, match="duplicate"):
        detection_layout("train", Split.TRAIN)


def test_classes():
    assert ExDark.classes is not None and DarkFace.classes is not None
    assert ExDark.classes.names[10] == "people" and ExDark.classes["table"].id == 11
    assert DarkFace.classes.names == ["face"]


@pytest.mark.parametrize("cls", CLASSES.values())
def test_loads_synthetic_folder(tmp_path: StdPath, cls: type):
    _write(tmp_path, "train", {"a": "0 0.5 0.5 0.5 0.5\n0 0.25 0.25 0.1 0.1\n", "b": ""})
    ds = cls("train", root=tmp_path)
    assert len(ds) == 2
    s = ds[0]
    assert set(s) == {"image", "boxes", "labels", "meta"}
    assert isinstance(s["boxes"], tv_tensors.BoundingBoxes) and s["boxes"].shape == (2, 4)
    assert s["boxes"][0].tolist() == [15.0, 10.0, 45.0, 30.0]  # XYXY pixels on 60×40
    assert s["labels"].tolist() == [0, 0]
    assert ds[1]["boxes"].shape == (0, 4)  # empty label file: no objects


# --- Real data (skipped unless the datasets are on this machine) ---

def _real_root(name: str) -> StdPath:
    try:
        root = datasets_dir() / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
@pytest.mark.parametrize("name", CLASSES)
def test_real_counts_and_boxes(name: str):
    root = _real_root(name)
    for split, (n_images, n_boxes) in EXPECTED[name].items():
        ds = CLASSES[name](split, root=root)
        assert len(ds) == n_images
        total = 0
        for _, paths in ds.items():
            total += sum(1 for line in paths["objects"].read_text().splitlines() if line.strip())
        assert total == n_boxes
        s = ds[0]
        assert s["image"].dtype == torch.uint8 and s["image"].shape[0] == 3
        assert int(s["labels"].max()) < len(CLASSES[name].classes)
