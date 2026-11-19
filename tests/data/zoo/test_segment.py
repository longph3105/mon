"""Tests for ``mon.data.zoo.segment``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.segment import CITYSCAPES_CLASSES, IGNORE_INDEX, NightCity, cityscapes_train_ids


def test_registered():
    assert DATASETS.get("nightcity") is NightCity
    assert DATASETS.entry("nightcity").meta["tasks"] == {Task.SEGMENT, Task.LLE}
    assert DATASETS.entry("nightcity").meta["splits"] == {Split.TRAIN, Split.VAL}


def test_cityscapes_classes_and_train_ids():
    assert len(CITYSCAPES_CLASSES) == 19
    assert [c.source_id for c in CITYSCAPES_CLASSES][:3] == [7, 8, 11]
    assert CITYSCAPES_CLASSES.by_id(18).name == "bicycle" and CITYSCAPES_CLASSES["sky"].color == (70, 130, 180)
    ids = torch.arange(256, dtype=torch.uint8)
    out = cityscapes_train_ids(ids)
    assert out.dtype == torch.uint8
    assert out[[7, 8, 26, 33]].tolist() == [0, 1, 13, 18]
    keep = {c.source_id for c in CITYSCAPES_CLASSES}
    assert all(int(out[i]) == IGNORE_INDEX for i in range(256) if i not in keep)


def _write(root: StdPath, split: str, mask: torch.Tensor) -> None:
    write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), root / split / "images" / "c_0001.png")
    write_image(mask[None], root / split / "labels" / "c_0001.png")


def test_loads_synthetic_and_maps_ids(tmp_path: StdPath):
    mask = torch.full((4, 6), 7, dtype=torch.uint8)
    mask[0, 0], mask[1, 1] = 0, 26
    _write(tmp_path, "train", mask)
    s = NightCity("train", root=tmp_path)[0]
    assert set(s) == {"image", "mask", "meta"}
    assert isinstance(s["mask"], tv_tensors.Mask) and s["mask"].dtype == torch.uint8
    assert int(s["mask"][0, 0]) == IGNORE_INDEX and int(s["mask"][1, 1]) == 13 and int(s["mask"][2, 2]) == 0


def test_double_size_mask_is_downsampled_nearest(tmp_path: StdPath):
    mask = torch.full((8, 12), 23, dtype=torch.uint8)  # 2x the 4x6 image, as NightLab's val labels
    mask[0:2, 0:2] = 26
    _write(tmp_path, "val", mask)
    ds = NightCity("val", root=tmp_path, transform=v2.Resize((2, 3), interpolation=v2.InterpolationMode.NEAREST))
    s = ds[0]
    assert s["mask"].shape == (2, 3) and s["image"].shape == (3, 2, 3)
    assert int(s["mask"][0, 0]) == 13 and int(s["mask"][1, 2]) == 10  # car, sky
    assert ds.transform is not None  # restored after __getitem__


@pytest.mark.slow
def test_real_counts_and_sizes():
    try:
        root = datasets_dir() / "nightcity"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    for split, n in {Split.TRAIN: 2998, Split.VAL: 1299}.items():
        ds = NightCity(split, root=root)
        assert len(ds) == n
        s = ds[0]
        assert s["image"].shape == (3, 512, 1024) and s["mask"].shape == (512, 1024)
        values = set(s["mask"].unique().tolist())
        assert values <= set(range(19)) | {IGNORE_INDEX}
