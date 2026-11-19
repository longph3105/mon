"""Tests for ``mon.data.zoo.track``: a synthetic folder, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.track import MOT16


def test_registered():
    assert DATASETS.get("mot16") is MOT16
    assert DATASETS.entry("mot16").meta["tasks"] == {Task.TRACK, Task.DETECT}
    assert DATASETS.entry("mot16").meta["splits"] == {Split.TRAIN, Split.TEST}
    assert MOT16.classes is not None and MOT16.classes.names == ["person"]


def test_loads_synthetic_folder(tmp_path: StdPath):
    frame = torch.zeros(3, 40, 60, dtype=torch.uint8)
    write_image(frame, tmp_path / "train" / "images" / "MOT16-02_000001.jpg")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "MOT16-02_000001.txt").write_text("0 0.5 0.5 0.2 0.5\n")
    write_image(frame, tmp_path / "test" / "seq" / "MOT16-01" / "img1" / "000001.jpg")
    (tmp_path / "test" / "seq" / "MOT16-01" / "seqinfo.ini").write_text("[Sequence]\n")  # not an image: ignored
    s = MOT16("train", root=tmp_path)[0]
    assert set(s) == {"image", "boxes", "labels", "meta"} and s["labels"].tolist() == [0]
    assert s["boxes"].tolist() == [[24.0, 10.0, 36.0, 30.0]]
    t = MOT16("test", root=tmp_path)
    assert len(t) == 1 and set(t[0]) == {"image", "meta"} and t[0]["meta"].stem == "MOT16-01/img1/000001"


@pytest.mark.slow
def test_real_counts_and_boxes():
    try:
        root = datasets_dir() / "mot16"
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    train = MOT16("train", root=root)
    assert len(train) == 5316
    rows = [line.split() for _, p in train.items() for line in p["objects"].read_text().splitlines() if line.strip()]
    assert len(rows) == 110405 and {r[0] for r in rows} == {"0"}
    for _, cx, cy, w, h in ((r[0], *map(float, r[1:])) for r in rows):  # clipped to the frame by make_mot16.sh
        assert 0 < w <= 1 and 0 < h <= 1 and cx - w / 2 > -1e-6 and cx + w / 2 < 1 + 1e-6
        assert cy - h / 2 > -1e-6 and cy + h / 2 < 1 + 1e-6
    assert len(MOT16("test", root=root)) == 5919
