"""Tests for ``mon.data.zoo.ydld``: a synthetic folder, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.ydld import YDLD

EXPECTED = {Split.TRAIN: (1588, 49384), Split.TEST: (1928, 66641)}


def test_registered():
    assert DATASETS.get("ydld") is YDLD and YDLD.folder == "ydld"
    assert DATASETS.entry("ydld").meta["tasks"] == {Task.DETECT, Task.LLE}
    assert DATASETS.entry("ydld").meta["splits"] == {Split.TRAIN, Split.TEST}
    assert YDLD.classes is not None
    assert YDLD.classes.names == ["streetlight", "carlight", "traffic_signal_light"]


def test_loads_synthetic_folder(tmp_path: StdPath):
    for stem, text in {"a": "0 0.5 0.5 0.5 0.5\n0 0.25 0.25 0.1 0.1\n", "b": ""}.items():
        write_image(torch.zeros(3, 40, 60, dtype=torch.uint8), tmp_path / "train" / "images" / f"{stem}.png")
        (tmp_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (tmp_path / "train" / "labels" / f"{stem}.txt").write_text(text)
    ds = YDLD("train", root=tmp_path)
    assert len(ds) == 2
    s = ds[0]
    assert set(s) == {"image", "boxes", "labels", "meta"}
    assert isinstance(s["boxes"], tv_tensors.BoundingBoxes) and s["boxes"].shape == (2, 4)
    assert s["boxes"][0].tolist() == [15.0, 10.0, 45.0, 30.0]  # XYXY pixels on 60×40
    assert s["labels"].tolist() == [0, 0]
    assert ds[1]["boxes"].shape == (0, 4)  # empty label file: no objects


# --- Real data (skipped unless the dataset is on this machine) ---

def _real_root() -> StdPath:
    try:
        root = datasets_dir() / "ydld"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
def test_real_counts_and_boxes():
    root = _real_root()
    for split, (n_images, n_boxes) in EXPECTED.items():
        ds = YDLD(split, root=root)
        assert len(ds) == n_images
        total = sum(sum(1 for line in p["objects"].read_text().splitlines() if line.strip()) for _, p in ds.items())
        assert total == n_boxes
        s = ds[0]
        assert s["image"].dtype == torch.uint8 and s["image"].shape[0] == 3
        assert YDLD.classes is not None and int(s["labels"].max()) < len(YDLD.classes)
