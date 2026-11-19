"""DARK FACE: face detection in low-light night scenes.

Stored in the normalized detection layout (``mon.data.dataset.detection_layout``; workspace
``data/README.md``): ``<split>/images/`` with YOLO labels in ``<split>/labels/`` (``class cx cy w h``,
normalized), class ids in ``classes.txt`` order.

Also tagged ``Task.LLE``: low-light images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-detection.

Warning:
    Upstream has no labeled test set (the 4,000 hold-out labels are withheld), so ``test`` is a
    seeded random hold-out of 1,000 of the 6,000 labeled images. Near-duplicate frames of one
    scene fall on both sides: test scores are optimistic and not comparable to the challenge
    leaderboard. Faces are tiny (often < 20 px).

Examples:
    >>> from mon.core.enums import Task
    >>> from mon.data.dataset import DATASETS
    >>> "darkface" in set(DATASETS.names(tasks=Task.DETECT)) & set(DATASETS.names(tasks=Task.LLE))
    True
    >>> DarkFace.classes.names
    ['face']

References:
    - Yang et al., "Advancing Image Understanding in Poor Visibility Environments: A Collective
      Benchmark Study", IEEE TIP 2020. https://flyywh.github.io/CVPRW2019LowLight/
    - Built by ``make_darkface.sh`` inside the dataset folder.
"""

__all__ = [
    "DarkFace",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("darkface", tasks={Task.DETECT, Task.LLE})
class DarkFace(FolderDataset):
    """DARK FACE: 5,000 train / 1,000 test PNGs (1080×720) of night scenes, one class (``face``), 50,396 boxes."""

    folder  = "darkface"
    layout  = detection_layout("train", "test")
    classes = ClassList.from_names(["face"])
