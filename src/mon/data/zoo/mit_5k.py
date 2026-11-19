"""MIT-Adobe FiveK: photo retouching / enhancement pairs (input → Expert C).

Paired 8-bit JPEGs, 768 px on the long side (sizes vary per photo), stored in the normalized paired
layout: ``<split>/images/`` = the input rendering, ``<split>/targets/`` = Expert C's retouch (the
usual FiveK target), named by the FiveK basename (``a0001-jmac_DSC1459.jpg``). The split is
yuukicammy/mit-adobe-fivek-dataset's ``training``/``validation``/``testing`` JSON (3,500 / 500 /
1,000), not the "first 4,500 / last 500" protocol of HDRNet/DeepUPE: numbers are not comparable
with papers using that split.

Warning:
    These are the owner's renderings of the official DNGs and expert TIFFs (768 px JPEGs; how they were
    rendered is not recorded), so absolute scores differ from papers' own renderings (e.g. 480p
    Lightroom exports). FiveK inputs are not systematically dark: it is a retouching benchmark,
    and SCI's "MIT" set (500 / 100, its own selection and rendering) is not reproduced here.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "mit_5k" in DATASETS.names(tasks=Task.LLE)
    True
    >>> MIT5K.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True
    >>> ds = DATASETS.build("mit_5k", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds)                                # doctest: +SKIP
    1000

References:
    - Bychkovsky et al., "Learning Photographic Global Tonal Adjustment with a Database of Input / Output
      Image Pairs", CVPR 2011. https://data.csail.mit.edu/graphics/fivek/ (research use)
    - Split: https://github.com/yuukicammy/mit-adobe-fivek-dataset (``data/*.json``).
    - Built by ``make_mit_5k.sh`` inside the dataset folder.
"""

__all__ = [
    "MIT5K",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, paired_layout, register_dataset


@register_dataset("mit_5k", tasks={Task.LLE})
class MIT5K(FolderDataset):
    """MIT-Adobe FiveK, input → Expert C: 3,500 train / 500 val / 1,000 test pairs."""

    folder = "mit_5k"
    layout = paired_layout("train", "val", "test")
