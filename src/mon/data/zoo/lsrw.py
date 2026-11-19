"""LSRW: real paired low-light / normal-light photos from R2RNet, one dataset per camera.

``lsrw_huawei`` (Huawei P40 Pro) and ``lsrw_nikon`` (Nikon D7500), each in the normalized paired
layout (``<split>/images/`` = low light, ``<split>/targets/`` = normal light) under ``lsrw/<camera>/``;
8-bit JPEGs, 960×640 (Nikon) and 960×720 or 960×540 (Huawei). Train / test: Huawei 2,450 / 30,
Nikon 3,150 / 20, together the paper's 5,600 / 50 (concatenate the two datasets for that).

Warning:
    47 Nikon train pairs appear twice under different names (input and target byte-identical, e.g.
    ``137``/``323``); kept as published. This is the owner's copy (the official release is only on
    Baidu Pan); it was not compared with it. No file is shared between train and test.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> {"lsrw_huawei", "lsrw_nikon"} <= set(DATASETS.names(tasks=Task.LLE))
    True
    >>> LSRWNikon.splits() == {Split.TRAIN, Split.TEST}
    True
    >>> ds = DATASETS.build("lsrw_huawei", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["image"].shape               # doctest: +SKIP
    (30, torch.Size([3, 720, 960]))

References:
    - Hai et al., "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network", JVCIR 2023.
      https://arxiv.org/abs/2106.14501, https://github.com/JianghaiSCU/R2RNet
    - Normalized by ``make_lsrw.sh`` inside the dataset folder.
"""

__all__ = [
    "LSRWHuawei",
    "LSRWNikon",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, paired_layout, register_dataset


@register_dataset("lsrw_huawei", tasks={Task.LLE})
class LSRWHuawei(FolderDataset):
    """LSRW, Huawei P40 Pro: 2,450 train and 30 test pairs."""

    folder = "lsrw/huawei"
    layout = paired_layout("train", "test")


@register_dataset("lsrw_nikon", tasks={Task.LLE})
class LSRWNikon(FolderDataset):
    """LSRW, Nikon D7500: 3,150 train and 20 test pairs."""

    folder = "lsrw/nikon"
    layout = paired_layout("train", "test")
