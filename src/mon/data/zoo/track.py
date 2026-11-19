"""Multi-object tracking datasets: MOT16.

MOT16 is stored as released (``<split>/seq/MOT16-XX/``: ``img1/<frame>.jpg``, ``det/det.txt``,
``seqinfo.ini``, and ``gt/gt.txt`` in train only) plus a YOLO detection view of train built by
``make_mot16.sh`` (workspace ``data/README.md``): ``train/images/<seq>_<frame>.jpg`` with
``train/labels/`` (``class cx cy w h``, normalized). Samples here are frames:

- ``train``: ``{"image", "boxes", "labels", "meta"}``, the boxes MOT evaluation counts (pedestrians
  with flag 1, clipped to the frame; class ``person``), ``meta.stem`` = ``"<seq>_<frame>"``.
- ``test``: ``{"image", "meta"}`` (no released ground truth), ``meta.stem`` = ``"<seq>/img1/<frame>"``.

Track identities, visibility and the public detections stay in ``seq/`` and are not loaded: this
is a per-frame view (detection, or per-frame inputs to a tracker), not a sequence loader.

Warning:
    Frames of one sequence are consecutive video frames (nearly identical neighbours): never split
    train by frame. 15,654 of the 110,405 train boxes are under 10 % visible (kept, as MOT
    evaluation counts them). Two sequences per split are 640×480 (``MOT16-05``, ``MOT16-06``);
    the rest are 1920×1080.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "mot16" in set(DATASETS.names(tasks=Task.TRACK)) & set(DATASETS.names(tasks=Task.DETECT))
    True
    >>> MOT16.splits() == {Split.TRAIN, Split.TEST}
    True
    >>> MOT16.classes.names
    ['person']
    >>> ds = DATASETS.build("mot16", "train")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["meta"].stem            # doctest: +SKIP
    (5316, 'MOT16-02_000001')

References:
    - Milan et al., "MOT16: A Benchmark for Multi-Object Tracking", arXiv 2016.
      https://arxiv.org/abs/1603.00831, https://motchallenge.net/data/MOT16/ (CC BY-NC-SA 3.0)
    - YOLO view built by ``make_mot16.sh`` inside the dataset folder.
"""

__all__ = [
    "MOT16",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, register_dataset
from mon.data.structures import ClassList
from mon.data.zoo.detect import detection_layout


@register_dataset("mot16", tasks={Task.TRACK, Task.DETECT})
class MOT16(FolderDataset):
    """MOT16: 5,316 train frames (7 sequences, 110,405 person boxes) and 5,919 test frames (7 sequences, no labels)."""

    folder  = "mot16"
    layout  = {
        **detection_layout("train"),
        Split.TEST: {"image": Modality("test/seq")},
    }
    classes = ClassList.from_names(["person"])
