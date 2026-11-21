"""Tests for ``tools/dataset_overview.py`` (synthetic datasets in ``tmp_path``)."""

import importlib.util
import sys
from pathlib import Path as StdPath
from types import ModuleType

import cv2
import numpy as np
import pytest
import torch

from mon.core.enums import Split
from mon.data.dataset import ClassFolderDataset, FolderDataset, Modality, classification_layout, detection_layout
from mon.data.io import write_image
from mon.data.structures import ClassList

TOOL = StdPath(__file__).parents[2] / "tools" / "dataset_overview.py"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dataset_overview", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["dataset_overview"] = module
    spec.loader.exec_module(module)
    return module


def img(value: int = 100, h: int = 40, w: int = 60) -> torch.Tensor:
    return torch.full((3, h, w), value, dtype=torch.uint8)


class Paired(FolderDataset):
    layout = {Split.TEST: {"image": Modality("test/images"), "target": Modality("test/targets")}}


class Sequence(FolderDataset):
    layout = {Split.TRAIN: {"image": Modality("train/images"), "target": Modality("train/targets")}}

    @classmethod
    def pair_key(cls, modality: str, key: str) -> str:
        return key.rsplit("/", 1)[0] if modality == "target" else key


class Det(FolderDataset):
    classes = ClassList.from_names(["car", "person"])
    layout  = detection_layout("val")


class Mot(FolderDataset):
    layout = {Split.TEST: {"image": Modality("test/seq")}}


class Cls(ClassFolderDataset):
    layout = classification_layout("test")


def test_sample_view_pairs(tool: ModuleType, tmp_path: StdPath):
    for k in range(3):
        write_image(img(20), tmp_path / f"test/images/{k}.png")
        write_image(img(200), tmp_path / f"test/targets/{k}.png")
    ds = Paired("test", root=tmp_path)
    assert not tool.is_sequence(ds) and not tool.is_mot_test(ds)
    out = tool.overview(ds, n=20, name="paired test")
    assert out.dtype == np.uint8 and out.ndim == 3 and out.shape[1] <= tool.MAX_W + 20
    assert tool.split_dir(ds) == tmp_path / "test"


def test_boxes_are_drawn_with_class_colors(tool: ModuleType, tmp_path: StdPath):
    write_image(img(0, 100, 100), tmp_path / "val/images/a.png")
    (tmp_path / "val/labels").mkdir(parents=True)
    (tmp_path / "val/labels/a.txt").write_text("1 0.5 0.5 0.5 0.5\n")
    ds  = Det("val", root=tmp_path)
    out = tool.overview(ds, n=1, name="det val")
    green, orange = np.array(tool.PALETTE[0]), np.array(tool.PALETTE[1])
    assert (out == orange).all(axis=-1).any()                    # class 1's color appears
    assert (out == green).all(axis=-1).sum() < 20 * 20 * 4       # class 0 only in the legend


def test_sequence_view(tool: ModuleType, tmp_path: StdPath):
    for scene in ("1", "2"):
        for k in range(1, 13):
            write_image(img(10 * k), tmp_path / f"train/images/{scene}/{k}.png")
        write_image(img(250), tmp_path / f"train/targets/{scene}.png")
    ds = Sequence("train", root=tmp_path)
    assert tool.is_sequence(ds)
    assert tool.overview(ds, n=20, max_seq=4).shape[0] > 0


def test_flat_shared_targets_are_not_sequences(tool: ModuleType, tmp_path: StdPath):
    class Shared(Paired):
        @classmethod
        def pair_key(cls, modality: str, key: str) -> str:
            return "t" if modality == "target" else key

    write_image(img(), tmp_path / "test/images/a.png")
    write_image(img(), tmp_path / "test/images/b.png")
    write_image(img(), tmp_path / "test/targets/t.png")
    assert not tool.is_sequence(Shared("test", root=tmp_path))


def test_mot_test_view(tool: ModuleType, tmp_path: StdPath):
    seq = tmp_path / "test/seq/MOT-01"
    for f in (1, 2):
        write_image(img(0, 100, 160), seq / f"img1/{f:06d}.jpg", quality=95)
    (seq / "det").mkdir()
    (seq / "det/det.txt").write_text("1,-1,10,10,30,60,0.9,-1,-1,-1\n2,-1,50,20,30,60,0.8,-1,-1,-1\n")
    ds = Mot("test", root=tmp_path)
    assert tool.is_mot_test(ds)
    out = tool.overview(ds, n=2)
    assert (out == np.array(tool.PALETTE[0])).all(axis=-1).any()


def test_class_folder_split_dir(tool: ModuleType, tmp_path: StdPath):
    (tmp_path / "classes.txt").write_text("a\nb\n")
    write_image(img(), tmp_path / "test/a/0.png")
    write_image(img(), tmp_path / "test/b/1.png")
    ds = Cls("test", root=tmp_path)
    assert tool.split_dir(ds) == tmp_path / "test"
    assert tool.overview(ds, n=2).shape[0] > 0


def test_write_overviews_folder(tool: ModuleType, tmp_path: StdPath):
    folder = tmp_path / "imgs"
    write_image(img(), folder / "x/a.png")
    write_image(img(), folder / "b.png")
    logs: list[str] = []
    assert tool.write_overviews(str(folder), log=logs.append) == [folder / "overview.jpg"]
    assert cv2.imread(str(folder / "overview.jpg")) is not None
    assert tool.write_overviews(str(folder), log=logs.append) == [] and logs[-1].startswith("skip")
    assert tool.write_overviews(str(folder), overwrite=True, log=logs.append) == [folder / "overview.jpg"]
    with pytest.raises(SystemExit):
        tool.write_overviews(str(tmp_path / "missing"))
