"""UHD-LL: ultra-high-definition (4K) paired low-light enhancement, from UHDFour.

Paired low-light / normal-light JPEGs (3840×2160, 8-bit sRGB), stored in the normalized paired
layout (``<split>/images/`` = low-light input, ``<split>/targets/`` = normal-light), with the
upstream names ``<n>_UHD_LL.JPG``.

Warning:
    Several low-light inputs share one target: 1,461 train targets fall in 402 groups of
    byte-identical files, and 82 test targets in 23 groups (the inputs are all distinct), so pairs
    aren't independent; never re-split randomly by image. No file is shared between train and test.
    At 4K most models need tiled inference (``mon.nn.tiling``) or a downsampled copy to fit in
    memory (see ``docs/efficiency.md``).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "uhd_ll" in DATASETS.names(tasks=Task.LLE)
    True
    >>> UHDLL.splits() == {Split.TRAIN, Split.TEST}
    True
    >>> ds = DATASETS.build("uhd_ll", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["image"].shape          # doctest: +SKIP
    (150, torch.Size([3, 2160, 3840]))

References:
    - Li et al., "Embedding Fourier for Ultra-High-Definition Low-Light Image Enhancement", ICLR 2023.
      https://arxiv.org/abs/2302.11831, https://github.com/Li-Chongyi/UHDFour_code
    - Built by ``make_uhd_ll.sh`` inside the dataset folder (from the official Google Drive release).
"""

__all__ = [
    "UHDLL",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, paired_layout, register_dataset


@register_dataset("uhd_ll", tasks={Task.LLE})
class UHDLL(FolderDataset):
    """UHD-LL: 2,000 train and 150 test 4K pairs (``training_set`` / ``testing_set``)."""

    folder = "uhd_ll"
    layout = paired_layout("train", "test")
