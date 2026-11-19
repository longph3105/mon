"""NTIRE challenge datasets: NTIRE 2025 Low Light Image Enhancement.

Stored as ``train/images/<id>.png`` + ``train/targets/<id>.png`` (paired) and ``val/images/``,
``test/images/`` (inputs only), with the challenge's image ids. Images are 2992×2000 or
6000×4000 8-bit PNGs. ``train`` samples are ``{"image", "target", "meta"}``; ``val`` and
``test`` are ``{"image", "meta"}``: their ground truth was never released, so score them with
no-reference metrics or through the challenge server. The challenge's scoring script is in
``scoring_program/`` next to the data.

Warning:
    Several scenes were shot more than once from nearby viewpoints, and val ``74`` is the same
    corridor as train ``73`` at another exposure; never re-split by image without grouping
    scenes. The train pairs come from a Hugging Face mirror (the organizers' copies need a
    login); their challenge ids follow the mirror's naming order, checked on all 76 val/test
    files.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "ntire_2025_llie" in DATASETS.names(tasks=Task.LLE)
    True
    >>> NTIRE2025LLIE.splits() == {Split.TRAIN, Split.VAL, Split.TEST}
    True
    >>> ds = DATASETS.build("ntire_2025_llie", "train")   # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["image"].shape                     # doctest: +SKIP
    (219, torch.Size([3, 2000, 2992]))

References:
    - Liu et al., "NTIRE 2025 Challenge on Low Light Image Enhancement: Methods and Results", CVPRW
      2025. https://arxiv.org/abs/2510.13670, https://codalab.lisn.upsaclay.fr/competitions/21636
    - Data: https://huggingface.co/datasets/Bubu0320/NTIRE_LLE_2025 (train), the challenge's
      ``Val_Input``/``Test_Input``. Built by ``make_ntire_2025_llie.sh`` inside the dataset folder.
"""

__all__ = [
    "NTIRE2025LLIE",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, paired_layout, register_dataset, unpaired_layout


@register_dataset("ntire_2025_llie", tasks={Task.LLE})
class NTIRE2025LLIE(FolderDataset):
    """NTIRE 2025 LLIE: 219 train pairs, 46 val and 30 test inputs (no released ground truth)."""

    folder = "ntire_2025_llie"
    layout = {
        **paired_layout("train"),
        **unpaired_layout("val", "test"),
    }
