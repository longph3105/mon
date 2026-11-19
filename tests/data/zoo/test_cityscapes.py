"""Tests for ``mon.data.zoo.cityscapes``: the 19 evaluated classes and the labelId → train id map."""

import torch

from mon.data.zoo.cityscapes import CITYSCAPES_CLASSES, IGNORE_INDEX, cityscapes_train_ids


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
