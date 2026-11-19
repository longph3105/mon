"""LoLI-Street: synthetic low-light street scenes at three darkness levels, plus a real low-light test set.

Stored in the multi-exposure layout (workspace ``data/README.md``): ``<split>/images/<n>/<level>.jpg``
with ``<level>`` in ``dense``, ``moderate``, ``light`` (darkest to lightest), one normal-light
target per scene in ``<split>/targets/<n>.jpg``, and ``test/images/Test_<k>.jpg`` (real dusk
frames, no targets). All images are 512×512 JPEGs. One sample per low-light image,
``{"image", "target", "meta"}`` (``test``: ``{"image", "meta"}``), with ``meta.stem`` =
``"<n>/<level>"``. The upstream YOLO labels (COCO-80, likely pseudo-labels) are stored next to the
images but not loaded.

Warning:
    The low-light images are synthetic (darkened daytime photos). Only 9,457 of the 10,000 train
    scenes are distinct (543 repeat other scenes byte-for-byte), and val scene ``30727`` is not
    darkened (every level equals the target); both are kept as published. The three levels of a
    scene share a target, so never re-split randomly by image. Consecutive ``test`` frames are
    near-identical. No overlap between train, val and test.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "loli_street" in DATASETS.names(tasks=Task.LLE)
    True
    >>> LoLIStreet.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True
    >>> LoLIStreet.pair_key("target", "1/dense")
    '1'
    >>> ds = DATASETS.build("loli_street", "val")   # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["meta"].stem                 # doctest: +SKIP
    (3000, '30001/dense')

References:
    - Islam et al., "LoLI-Street: Benchmarking Low-Light Image Enhancement and Beyond", ACCV 2024.
      https://arxiv.org/abs/2410.09831, https://github.com/tanvirnwu/TriFuse
    - Data: https://www.kaggle.com/datasets/tanvirnwu/loli-street-low-light-image-enhancement-of-street
      (CC BY-NC 4.0). Built by ``make_loli_street.sh`` inside the dataset folder.
"""

__all__ = [
    "LoLIStreet",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, paired_layout, register_dataset


@register_dataset("loli_street", tasks={Task.LLE})
class LoLIStreet(FolderDataset):
    """LoLI-Street: 30,000 train / 3,000 val synthetic images (10,000 / 1,000 scenes), 1,000 real test images."""

    folder = "loli_street"
    layout = {
        **paired_layout("train", "val"),
        Split.TEST: {"image": Modality("test/images")},
    }

    @classmethod
    def pair_key(cls, modality: str, key: str) -> str:
        """Low-light image ``"<n>/<level>"`` → target ``"<n>"``."""
        return key.split("/", 1)[0] if modality == "target" else key
