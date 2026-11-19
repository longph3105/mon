"""Tests for ``mon.data.zoo.track``: a synthetic folder, plus checks on the real data when present."""

from pathlib import Path as StdPath

import pytest
import torch

from mon.core.enums import Split, Task
from mon.core.path import datasets_dir
from mon.data.dataset import DATASETS
from mon.data.io import write_image
from mon.data.zoo.track import MOT16, MOT20


CLASSES = {"mot16": MOT16, "mot20": MOT20}
EXPECTED = {"mot16": (5316, 110405, 5919), "mot20": (8931, 1134614, 4479)}  # train frames, train boxes, test frames


@pytest.mark.parametrize(("name", "cls"), CLASSES.items())
def test_registered(name: str, cls: type):
    assert DATASETS.get(name) is cls
    assert DATASETS.entry(name).meta["tasks"] == {Task.TRACK, Task.DETECT}
    assert DATASETS.entry(name).meta["splits"] == {Split.TRAIN, Split.TEST}
    assert cls.classes is not None and cls.classes.names == ["person"]


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
@pytest.mark.parametrize("name", CLASSES)
def test_real_counts_and_boxes(name: str):
    try:
        root = datasets_dir() / name
    except RuntimeError:
        pytest.skip("DATA_DIR / DATASETS_DIR not set")
    if not root.is_dir():
        pytest.skip(f"{root} not on this machine")
    n_train, n_boxes, n_test = EXPECTED[name]
    train = CLASSES[name]("train", root=root)
    assert len(train) == n_train
    rows = [line.split() for _, p in train.items() for line in p["objects"].read_text().splitlines() if line.strip()]
    assert len(rows) == n_boxes and {r[0] for r in rows} == {"0"}
    for _, cx, cy, w, h in ((r[0], *map(float, r[1:])) for r in rows):  # clipped to the frame by make_mot16.sh
        assert 0 < w <= 1 and 0 < h <= 1 and cx - w / 2 > -1e-6 and cx + w / 2 < 1 + 1e-6
        assert cy - h / 2 > -1e-6 and cy + h / 2 < 1 + 1e-6
    assert len(CLASSES[name]("test", root=root)) == n_test
