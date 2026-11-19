"""Low-light object detection datasets: ExDark and DARK FACE.

Also tagged ``Task.LLE``: low-light images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-detection.

Stored in the normalized detection layout (workspace ``data/README.md``): ``<split>/images/``
with YOLO labels in ``<split>/labels/`` (``class cx cy w h``, normalized) and a COCO
``<split>/annotations.json`` next to them (not read here). Samples are
``{"image", "boxes", "labels", "meta"}``: ``boxes`` in pixels (``XYXY``), ``labels`` the class
ids of ``classes`` (0-based, ``classes.txt`` order). An image without boxes has an empty label
file and yields empty ``boxes``.

Warning:
    - ExDark: 18 byte-identical image pairs, 3 across train/val and 2 across val/test (none
      across train/test); the bbGt annotations are noisy (e.g. a ``people`` box on a drink
      carton in ``2015_04665``). 37 images are grayscale JPEGs (read as RGB).
    - DARK FACE: upstream has no labeled test set (the 4,000 hold-out labels are withheld), so
      ``test`` is a seeded random hold-out of 1,000 of the 6,000 labeled images. Near-duplicate
      frames of one scene fall on both sides: test scores are optimistic and not comparable to
      the challenge leaderboard. Faces are tiny (often < 20 px).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> {"darkface", "exdark"} <= set(DATASETS.names(tasks=Task.DETECT)) & set(DATASETS.names(tasks=Task.LLE))
    True
    >>> ExDark.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True
    >>> ExDark.classes.names[:3], len(ExDark.classes), DarkFace.classes.names
    (['bicycle', 'boat', 'bottle'], 12, ['face'])
    >>> ds = DATASETS.build("exdark", "test")      # needs DATA_DIR  # doctest: +SKIP
    >>> s = ds[0]                                  # doctest: +SKIP
    >>> s["boxes"].shape, s["labels"].dtype        # doctest: +SKIP
    (torch.Size([2, 4]), torch.int64)

References:
    - ExDark: Loh & Chan, "Getting to Know Low-light Images with The Exclusively Dark Dataset",
      CVIU 2019. https://github.com/cs-chan/Exclusively-Dark-Image-Dataset (split:
      ``Groundtruth/imageclasslist.txt``, 250 train / 150 val / rest test per class).
    - DARK FACE: Yang et al., "Advancing Image Understanding in Poor Visibility Environments: A
      Collective Benchmark Study", IEEE TIP 2020. https://flyywh.github.io/CVPRW2019LowLight/
    - Built by ``make_exdark.sh`` / ``make_darkface.sh`` inside each dataset folder.
"""

__all__ = [
    "DarkFace",
    "ExDark",
    "detection_layout",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, register_dataset
from mon.data.structures import ClassList


def detection_layout(*splits: Split | str) -> dict[Split, dict[str, Modality]]:
    """Layout for detection datasets: ``<split>/images/`` → ``image``, ``<split>/labels/`` (YOLO) → boxes.

    Examples:
        >>> layout = detection_layout("train", "test")
        >>> layout[Split.TEST]["objects"].dir, layout[Split.TEST]["objects"].kind
        ('test/labels', 'yolo')
    """
    if not splits:
        raise ValueError("detection_layout needs at least one split")
    parsed = [Split(s) for s in splits]
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"duplicate splits: {[s.value for s in parsed]}")
    return {
        s: {"image": Modality(f"{s.value}/images"), "objects": Modality(f"{s.value}/labels", kind="yolo")}
        for s in parsed
    }


@register_dataset("exdark", tasks={Task.DETECT, Task.LLE})
class ExDark(FolderDataset):
    """ExDark: 3,000 train / 1,800 val / 2,563 test low-light JPEGs, 12 classes, 23,710 boxes (official split)."""

    folder  = "exdark"
    layout  = detection_layout("train", "val", "test")
    classes = ClassList.from_names([
        "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cup", "dog", "motorbike", "people", "table",
    ])


@register_dataset("darkface", tasks={Task.DETECT, Task.LLE})
class DarkFace(FolderDataset):
    """DARK FACE: 5,000 train / 1,000 test PNGs (1080×720) of night scenes, one class (``face``), 50,396 boxes."""

    folder  = "darkface"
    layout  = detection_layout("train", "test")
    classes = ClassList.from_names(["face"])
