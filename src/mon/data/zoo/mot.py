"""Multi-object tracking datasets: MOT16 and MOT20 (MOTChallenge).

Both are stored as released (``<split>/seq/<seq>/``: ``img1/<frame>.jpg``, ``det/det.txt``,
``seqinfo.ini``, and ``gt/gt.txt`` in train only) plus a YOLO detection view of train built by
``make_mot16.sh`` / ``make_mot20.sh`` (workspace ``data/README.md``): ``train/images/<seq>_<frame>.jpg``
with ``train/labels/`` (``class cx cy w h``, normalized). Samples here are frames:

- ``train``: ``{"image", "boxes", "labels", "meta"}``, the boxes MOT evaluation counts (pedestrians
  with flag 1, clipped to the frame; class ``person``), ``meta.stem`` = ``"<seq>_<frame>"``.
- ``test``: ``{"image", "meta"}`` (no released ground truth), ``meta.stem`` = ``"<seq>/img1/<frame>"``.

Track identities, visibility and the public detections stay in ``seq/`` and are not loaded: this
is a per-frame view (detection, or per-frame inputs to a tracker), not a sequence loader.

Warning:
    Frames of one sequence are consecutive video frames (nearly identical neighbours): never split
    train by frame. Heavily occluded boxes are kept, as MOT evaluation counts them (under 10 %
    visible: MOT16 15,654 of 110,405, MOT20 88,119 of 1,134,614). Frame sizes differ per sequence
    (MOT16: 1920×1080, ``MOT16-05``/``-06`` 640×480; MOT20: 1920×1080, 1173×880, 1654×1080,
    1545×1080, 1920×734). MOT20 is crowded: 127 boxes per train frame on average, 220 at most.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> {"mot16", "mot20"} <= set(DATASETS.names(tasks=Task.TRACK)) & set(DATASETS.names(tasks=Task.DETECT))
    True
    >>> MOT16.splits() == {Split.TRAIN, Split.TEST}
    True
    >>> MOT16.classes.names, MOT20.classes.names
    (['person'], ['person'])
    >>> ds = DATASETS.build("mot16", "train")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["meta"].stem            # doctest: +SKIP
    (5316, 'MOT16-02_000001')

References:
    - Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv 2016.
      https://arxiv.org/abs/1603.00831, https://motchallenge.net/data/MOT16/ (CC BY-NC-SA 3.0)
    - Dendorfer et al., "MOT20: A benchmark for multi object tracking in crowded scenes", arXiv 2020.
      https://arxiv.org/abs/2003.09003, https://motchallenge.net/data/MOT20/ (CC BY-NC-SA 3.0)
    - YOLO views built by ``make_mot16.sh`` / ``make_mot20.sh`` inside each dataset folder.
"""

__all__ = [
    "MOT16",
    "MOT20",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, detection_layout, register_dataset
from mon.data.structures import ClassList


@register_dataset("mot16", tasks={Task.TRACK, Task.DETECT})
class MOT16(FolderDataset):
    """MOT16: 5,316 train frames (7 sequences, 110,405 person boxes) and 5,919 test frames (7 sequences, no labels)."""

    folder  = "mot16"
    layout  = {
        **detection_layout("train"),
        Split.TEST: {"image": Modality("test/seq")},
    }
    classes = ClassList.from_names(["person"])


@register_dataset("mot20", tasks={Task.TRACK, Task.DETECT})
class MOT20(FolderDataset):
    """MOT20: 8,931 train frames (4 sequences, 1,134,614 person boxes), 4,479 test frames (4 sequences, no labels)."""

    folder  = "mot20"
    layout  = {
        **detection_layout("train"),
        Split.TEST: {"image": Modality("test/seq")},
    }
    classes = ClassList.from_names(["person"])
