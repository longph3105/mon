"""FishEye8K: road object detection from fisheye traffic cameras.

Stored in the normalized detection layout (``mon.data.dataset.detection_layout``; workspace
``data/README.md``): ``<split>/images/`` (JPEG) with YOLO labels in ``<split>/labels/`` (``class cx cy
w h``, normalized), class ids in ``classes.txt`` order. Splits follow the AI City Challenge 2024
Track 4 release, by camera with none shared: ``train`` is the 2023 train set (cameras 3, 5, 6, 8–18),
``val`` the 2023 test set (cameras 1, 2, 4, 7), ``test`` FishEye1Keval (cameras 19–29).

Warning:
    ``val`` holds 165 byte-identical image pairs (``camera1_A_i`` = ``camera1_A_{i+200}`` for
    i < 100, and 65 ``camera4_M`` frames) that upstream labelled twice, differently; they are kept
    so scores stay comparable with published results. ``test`` uses the revised labels
    (``labels_02``, 40,334 boxes); the first release is kept in ``test/labels_v1/``. 142 night/IR
    frames are single-channel JPEGs (``read_image`` expands them to RGB).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> DATASETS.entry("fisheye8k").meta["tasks"] == {Task.DETECT}
    True
    >>> FishEye8K.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True

References:
    - Gochoo et al., "FishEye8K: A Benchmark and Dataset for Fisheye Camera Object Detection", CVPRW
      2023. https://github.com/MoyoG/FishEye8K
    - Built by ``make_fisheye8k.sh`` inside the dataset folder (from the AI City 2024 Track 4 release).
"""

__all__ = [
    "FishEye8K",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("fisheye8k", tasks={Task.DETECT})
class FishEye8K(FolderDataset):
    """FishEye8K: 5,288 train / 2,712 val / 1,000 test images (112,205 / 45,145 / 40,334 boxes), 5 classes."""

    folder  = "fisheye8k"
    layout  = detection_layout("train", "val", "test")
    classes = ClassList.from_names(["Bus", "Bike", "Car", "Pedestrian", "Truck"])
