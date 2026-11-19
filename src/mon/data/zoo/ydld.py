"""YDLD: light-source detection in night driving videos (3 classes).

Stored in the normalized detection layout (``mon.data.dataset.detection_layout``; workspace
``data/README.md``): ``<split>/images/`` with YOLO labels in ``<split>/labels/`` (``class cx cy w h``,
normalized), class ids in ``classes.txt`` order.

Also tagged ``Task.LLE``: low-light images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-detection.

Warning:
    Lights are tiny (median short side 4.3 px at 720p); 10 test images are near-copies of train
    frames. ``<split>/annotations.json`` is the released COCO file (0-based category ids; 481
    ``area`` values differ from w·h, which COCO's size ranges use); the YOLO labels drop 3 zero-size
    train boxes. The full set is not public (the repository offers 100 samples).

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> "ydld" in set(DATASETS.names(tasks=Task.DETECT)) & set(DATASETS.names(tasks=Task.LLE))
    True
    >>> YDLD.classes.names
    ['streetlight', 'carlight', 'traffic_signal_light']

References:
    - "A New Multi-Source Light Detection Benchmark and Semi-Supervised Focal Light Detection",
      NeurIPS 2024 Datasets and Benchmarks. https://github.com/YDLD-dataset/YDLD
    - Built by ``make_ydld.sh`` inside the dataset folder.
"""

__all__ = [
    "YDLD",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("ydld", tasks={Task.DETECT, Task.LLE})
class YDLD(FolderDataset):
    """YDLD: 1,588 train / 1,928 test night driving frames, 3 light classes, 116,025 boxes (upstream split)."""

    folder  = "ydld"
    layout  = detection_layout("train", "test")
    classes = ClassList.from_names(["streetlight", "carlight", "traffic_signal_light"])
