"""LOL low-light enhancement datasets: LOL-v1 and LOL-v2-Real.

Paired low-light / normal-light PNG photos (600×400, 8-bit sRGB), stored in the normalized
paired layout (``<split>/images/`` = low-light input, ``<split>/targets/`` = normal-light).

Warning:
    LOL-v2-Real reuses LOL-v1 captures: all 15 LOL-v1 test inputs are in LOL-v2-Real (12 in its
    train split), and 86 LOL-v1 train inputs are in its test split (byte-identical files). Train
    and test within one dataset; never evaluate a model trained on one of them on the other.
    Several inputs of one scene share a target, so never re-split these datasets randomly.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> {"lol_v1", "lol_v2_real"} <= set(DATASETS.names(tasks=Task.LLE))
    True
    >>> LOLv1.splits() == {Split.TRAIN, Split.TEST}
    True
    >>> ds = DATASETS.build("lol_v1", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["image"].shape          # doctest: +SKIP
    (15, torch.Size([3, 400, 600]))

References:
    - LOL-v1: Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018.
      https://arxiv.org/abs/1808.04560, https://daooshee.github.io/BMVC2018website/
    - LOL-v2: Yang et al., "Sparse Gradient Regularized Deep Retinex Network for Robust Low-Light
      Image Enhancement", IEEE TIP 2021. https://github.com/flyywh/CVPR-2020-Semi-Low-Light
"""

__all__ = [
    "LOLv1",
    "LOLv2Real",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, paired_layout, register_dataset


@register_dataset("lol_v1", tasks={Task.LLE})
class LOLv1(FolderDataset):
    """LOL-v1: 485 train (``our485``) and 15 test (``eval15``) pairs."""

    folder = "lol_v1"
    layout = paired_layout("train", "test")


@register_dataset("lol_v2_real", tasks={Task.LLE})
class LOLv2Real(FolderDataset):
    """LOL-v2-Real (``Real_captured``): 689 train and 100 test pairs."""

    folder = "lol_v2_real"
    layout = paired_layout("train", "test")
