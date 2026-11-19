"""Tests for ``mon.data.zoo.mit_5k``: a synthetic folder, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.mit_5k import MIT5K

EXPECTED = {Split.TRAIN: 3500, Split.VAL: 500, Split.TEST: 1000}


def test_registered():
    assert DATASETS.get("mit_5k") is MIT5K
    assert DATASETS.entry("mit_5k").meta["tasks"] == {Task.LLE}
    assert DATASETS.entry("mit_5k").meta["splits"] == set(EXPECTED)


def test_loads_synthetic_folder(tmp_path: StdPath):
    for sub, value in (("images", 30), ("targets", 120)):
        write_image(torch.full((3, 4, 6), value, dtype=torch.uint8), tmp_path / "val" / sub / "a0001-jmac_DSC1459.jpg")
    s = MIT5K("val", root=tmp_path)[0]
    assert set(s) == {"image", "target", "meta"} and s["meta"].stem == "a0001-jmac_DSC1459"
    assert s["image"].float().mean() < s["target"].float().mean()


@pytest.mark.slow
def test_real_counts():
    try:
        root = datasets_dir() / "mit_5k"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    stems: set[str] = set()
    for split, n in EXPECTED.items():
        ds = MIT5K(split, root=root)
        assert len(ds) == n
        split_stems = {stem for stem, _ in ds.items()}
        assert not stems & split_stems  # disjoint splits
        stems |= split_stems
    assert len(stems) == 5000
    s = MIT5K("test", root=root)[0]
    assert s["image"].shape == s["target"].shape and max(s["image"].shape[1:]) == 768
