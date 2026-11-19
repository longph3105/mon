"""Tests for ``mon.data.zoo.ntire``: synthetic folders, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.ntire import NTIRE2025LLIE

EXPECTED = {Split.TRAIN: 219, Split.VAL: 46, Split.TEST: 30}


def test_registered():
    assert DATASETS.get("ntire_2025_llie") is NTIRE2025LLIE
    assert DATASETS.entry("ntire_2025_llie").meta["tasks"] == {Task.LLE}
    assert DATASETS.entry("ntire_2025_llie").meta["splits"] == set(EXPECTED)


def test_loads_synthetic_folder(tmp_path: StdPath):
    for sub, value in (("images", 1), ("targets", 2)):
        write_image(torch.full((3, 4, 6), value, dtype=torch.uint8), tmp_path / "train" / sub / "7.png")
    write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), tmp_path / "val" / "images" / "11.png")
    write_image(torch.zeros(3, 4, 6, dtype=torch.uint8), tmp_path / "test" / "images" / "108.png")
    s = NTIRE2025LLIE("train", root=tmp_path)[0]
    assert set(s) == {"image", "target", "meta"} and int(s["target"][0, 0, 0]) == 2 and s["meta"].stem == "7"
    for split in ("val", "test"):
        ds = NTIRE2025LLIE(split, root=tmp_path)
        assert len(ds) == 1 and set(ds[0]) == {"image", "meta"}


@pytest.mark.slow
def test_real_counts():
    try:
        root = datasets_dir() / "ntire_2025_llie"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    for split, n in EXPECTED.items():
        ds = NTIRE2025LLIE(split, root=root)
        assert len(ds) == n
        assert all(p["image"].suffix == ".png" for _, p in ds.items())
    s = NTIRE2025LLIE("train", root=root)[0]
    assert s["image"].shape == s["target"].shape and s["image"].shape[0] == 3
