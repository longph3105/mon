"""Tests for ``mon.data.zoo.uhd_ll``: a synthetic folder, plus checks on the real data when present."""

import hashlib
from collections import Counter
from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.uhd_ll import UHDLL

EXPECTED = {Split.TRAIN: 2000, Split.TEST: 150}


def test_registered():
    assert DATASETS.get("uhd_ll") is UHDLL
    assert DATASETS.entry("uhd_ll").meta["tasks"] == {Task.LLE}
    assert DATASETS.entry("uhd_ll").meta["splits"] == set(EXPECTED)


def test_loads_synthetic_folder(tmp_path: StdPath):
    for split in ("train", "test"):
        write_image(torch.full((3, 4, 6), 10, dtype=torch.uint8), tmp_path / split / "images" / "7_UHD_LL.JPG")
        write_image(torch.full((3, 4, 6), 200, dtype=torch.uint8), tmp_path / split / "targets" / "7_UHD_LL.JPG")
    s = UHDLL("test", root=tmp_path)[0]
    assert set(s) == {"image", "target", "meta"} and s["meta"].stem == "7_UHD_LL"
    assert s["image"].float().mean() < 50 < s["target"].float().mean()


@pytest.mark.slow
def test_real_counts_and_shared_targets():
    try:
        root = datasets_dir() / "uhd_ll"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    groups = {Split.TRAIN: (402, 1461), Split.TEST: (23, 82)}  # byte-identical target groups, files in them
    for split, n in EXPECTED.items():
        ds = UHDLL(split, root=root)
        assert len(ds) == n
        digests = Counter(hashlib.md5(p["target"].read_bytes()).hexdigest() for _, p in ds.items())
        shared = [c for c in digests.values() if c > 1]
        assert (len(shared), sum(shared)) == groups[split]
    s = UHDLL("test", root=root)[0]
    assert s["image"].shape == s["target"].shape == (3, 2160, 3840)
