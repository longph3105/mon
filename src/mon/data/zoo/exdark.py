"""ExDark: low-light object detection (12 classes), the official split.

Stored in the normalized detection layout (``mon.data.dataset.detection_layout``; workspace
``data/README.md``): ``<split>/images/`` with YOLO labels in ``<split>/labels/`` (``class cx cy w h``,
normalized), class ids in ``classes.txt`` order.

Also tagged ``Task.LLE``: low-light images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-detection.

Warning:
    18 byte-identical image pairs, 3 across train/val and 2 across val/test (none across
    train/test); the bbGt annotations are noisy (e.g. a ``people`` box on a drink carton in
    ``2015_04665``). 37 images are grayscale JPEGs (read as RGB).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "exdark" in set(DATASETS.names(tasks=Task.DETECT)) & set(DATASETS.names(tasks=Task.LLE))
    True
    >>> ExDark.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True
    >>> ExDark.classes.names[:3], len(ExDark.classes)
    (['bicycle', 'boat', 'bottle'], 12)
    >>> ds = DATASETS.build("exdark", "test")      # needs DATA_DIR  # doctest: +SKIP
    >>> s = ds[0]                                  # doctest: +SKIP
    >>> s["boxes"].shape, s["labels"].dtype        # doctest: +SKIP
    (torch.Size([2, 4]), torch.int64)

References:
    - Loh & Chan, "Getting to Know Low-light Images with The Exclusively Dark Dataset", CVIU 2019.
      https://github.com/cs-chan/Exclusively-Dark-Image-Dataset (split: ``Groundtruth/imageclasslist.txt``,
      250 train / 150 val / rest test per class).
    - Built by ``make_exdark.sh`` inside the dataset folder.
"""

__all__ = [
    "ExDark",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("exdark", tasks={Task.DETECT, Task.LLE})
class ExDark(FolderDataset):
    """ExDark: 3,000 train / 1,800 val / 2,563 test low-light JPEGs, 12 classes, 23,710 boxes (official split)."""

    folder  = "exdark"
    layout  = detection_layout("train", "val", "test")
    classes = ClassList.from_names([
        "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorbike", "people", "table",
    ])
