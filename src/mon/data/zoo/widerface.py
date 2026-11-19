"""WIDER FACE: face detection in everyday scenes.

Stored in the normalized detection layout (``mon.data.dataset.detection_layout``; workspace
``data/README.md``): ``<split>/images/`` with YOLO labels in ``<split>/labels/`` (``class cx cy w h``,
normalized), class ids in ``classes.txt`` order.

Warning:
    The YOLO labels keep every official box, invalid-flagged ones included (2,399 train, 585 val),
    except 38 with zero width or height; 4 train and 4 val images therefore have no box. The
    official evaluation's easy/medium/hard subsets (``.mat`` files in ``grouped.zip``) are not
    loaded, and ``test`` has no released labels (``{"image", "meta"}`` samples).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> DATASETS.entry("widerface").meta["tasks"] == {Task.DETECT}
    True
    >>> WiderFace.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True

References:
    - Yang et al., "WIDER FACE: A Face Detection Benchmark", CVPR 2016. http://shuoyang1213.me/WIDERFACE/
    - Built by ``make_widerface.sh`` inside the dataset folder (from the official ``grouped.zip``).
"""

__all__ = [
    "WiderFace",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("widerface", tasks={Task.DETECT})
class WiderFace(FolderDataset):
    """WIDER FACE: 12,880 train / 3,226 val images (159,393 / 39,697 face boxes), 16,097 test images (no labels)."""

    folder  = "widerface"
    layout  = {
        **detection_layout("train", "val"),
        Split.TEST: {"image": Modality("test/images")},
    }
    classes = ClassList.from_names(["face"])
