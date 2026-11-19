"""Tests for ``mon.data.zoo.fisheye8k``: a synthetic folder, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.fisheye8k import FishEye8K

EXPECTED = {Split.TRAIN: (5288, 112205), Split.VAL: (2712, 45145), Split.TEST: (1000, 40334)}


def test_registered():
    assert DATASETS.get("fisheye8k") is FishEye8K and FishEye8K.folder == "fisheye8k"
    assert DATASETS.entry("fisheye8k").meta["tasks"] == {Task.DETECT}
    assert DATASETS.entry("fisheye8k").meta["splits"] == {Split.TRAIN, Split.VAL, Split.TEST}
    assert FishEye8K.classes is not None
    assert FishEye8K.classes.names == ["Bus", "Bike", "Car", "Pedestrian", "Truck"]


def test_loads_synthetic_folder(tmp_path: StdPath):
    write_image(torch.zeros(3, 40, 60, dtype=torch.uint8), tmp_path / "test" / "images" / "camera19_A_0.jpg")
    (tmp_path / "test" / "labels").mkdir(parents=True)
    (tmp_path / "test" / "labels" / "camera19_A_0.txt").write_text("4 0.5 0.5 0.5 0.5\n1 0.25 0.25 0.1 0.1\n")
    ds = FishEye8K("test", root=tmp_path)
    assert len(ds) == 1
    s = ds[0]
    assert set(s) == {"image", "boxes", "labels", "meta"}
    assert isinstance(s["boxes"], tv_tensors.BoundingBoxes) and s["boxes"].shape == (2, 4)
    assert s["boxes"][0].tolist() == [15.0, 10.0, 45.0, 30.0]  # XYXY pixels on 60×40
    assert s["labels"].tolist() == [4, 1]


# --- Real data (skipped unless the dataset is on this machine) ---

def _real_root() -> StdPath:
    try:
        root = datasets_dir() / "fisheye8k"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    return root


@pytest.mark.slow
def test_real_counts_and_boxes():
    root = _real_root()
    for split, (n_images, n_boxes) in EXPECTED.items():
        ds = FishEye8K(split, root=root)
        assert len(ds) == n_images
        total = sum(sum(1 for line in p["objects"].read_text().splitlines() if line.strip()) for _, p in ds.items())
        assert total == n_boxes
        s = ds[0]
        assert s["image"].dtype == torch.uint8 and s["image"].shape[0] == 3
        assert FishEye8K.classes is not None and int(s["labels"].max()) < len(FishEye8K.classes)


@pytest.mark.slow
def test_real_gray_frame_loads_as_rgb():
    ds = FishEye8K("test", root=_real_root())
    i = next(i for i, (_, p) in enumerate(ds.items()) if p["image"].name.startswith("camera29_N"))
    assert ds[i]["image"].shape[0] == 3
