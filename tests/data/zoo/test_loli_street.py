"""Tests for ``mon.data.zoo.loli_street``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.loli_street import LoLIStreet

LEVELS = ("dense", "moderate", "light")


def test_registered():
    assert DATASETS.get("loli_street") is LoLIStreet
    assert DATASETS.entry("loli_street").meta["tasks"] == {Task.LLE}
    assert DATASETS.entry("loli_street").meta["splits"] == {Split.TRAIN, Split.VAL, Split.TEST}


def test_loads_synthetic_folder(tmp_path: StdPath):
    for n, value in (("1", 10), ("2", 20)):
        write_image(torch.full((3, 4, 6), value, dtype=torch.uint8), tmp_path / "train" / "targets" / f"{n}.jpg")
        for i, level in enumerate(LEVELS):
            write_image(torch.full((3, 4, 6), i, dtype=torch.uint8), tmp_path / "train" / "images" / n / f"{level}.jpg")
    write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), tmp_path / "test" / "images" / "Test_1.jpg")
    ds = LoLIStreet("train", root=tmp_path)
    assert len(ds) == 6
    stems = [stem for stem, _ in ds.items()]
    assert stems[:3] == ["1/dense", "1/light", "1/moderate"]
    assert all(paths["target"].name == f"{stem.split('/')[0]}.jpg" for stem, paths in ds.items())
    s = ds[0]
    assert set(s) == {"image", "target", "meta"} and abs(int(s["target"][0, 0, 0]) - 10) <= 2  # JPEG
    t = LoLIStreet("test", root=tmp_path)
    assert len(t) == 1 and set(t[0]) == {"image", "meta"}


@pytest.mark.slow
def test_real_counts():
    try:
        root = datasets_dir() / "loli_street"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    for split, n in {Split.TRAIN: 30000, Split.VAL: 3000, Split.TEST: 1000}.items():
        ds = LoLIStreet(split, root=root)
        assert len(ds) == n
        s = ds[0]
        assert s["image"].shape == (3, 512, 512)
        if split is not Split.TEST:
            assert s["target"].shape == (3, 512, 512)
