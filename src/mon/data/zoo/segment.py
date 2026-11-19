"""Night-time semantic segmentation datasets: NightCity (with NightLab's re-annotated val).

Also tagged ``Task.LLE``: night images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-segmentation.

Stored in the normalized segmentation layout (workspace ``data/README.md``): ``<split>/images/``
and single-channel PNG class maps in ``<split>/labels/``, with Cityscapes ``labelId`` values.
Samples are ``{"image", "mask", "meta"}``; ``mask`` holds Cityscapes **train ids** (0–18, the
19 evaluated classes of ``classes``; every other ``labelId`` and 255 → ``ignore_index`` 255).

Warning:
    NightCity's ``val/labels`` is NightLab's re-annotation (NightCity+), which recent papers
    evaluate on; it is stored at 2048×1024, twice the image size, and is downsampled here with
    nearest neighbour to the image size. The original NightCity val labels are in
    ``val/labels_nightcity`` (same size as the images): subclass and point the val ``mask`` there
    to use them. City prefixes keep upstream case (``tokyo`` and ``Tokyo``).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> "nightcity" in set(DATASETS.names(tasks=Task.SEGMENT)) & set(DATASETS.names(tasks=Task.LLE))
    True
    >>> NightCity.splits() == {Split.TRAIN, Split.VAL}
    True
    >>> len(NightCity.classes), NightCity.classes.by_id(0).name, NightCity.classes["road"].source_id
    (19, 'road', 7)
    >>> ds = DATASETS.build("nightcity", "val")   # needs DATA_DIR  # doctest: +SKIP
    >>> s = ds[0]                                 # doctest: +SKIP
    >>> s["image"].shape, s["mask"].shape         # doctest: +SKIP
    (torch.Size([3, 512, 1024]), torch.Size([512, 1024]))

References:
    - NightCity: Tan et al., "Night-time Scene Parsing with a Large Real Dataset", IEEE TIP 2021.
      https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html
    - NightLab (re-annotated val): Deng et al., "NightLab: A Dual-level Architecture with Hardness
      Detection for Segmentation at Night", CVPR 2022. https://github.com/xdeng7/NightLab
    - Built by ``make_nightcity.sh`` inside the dataset folder.
"""

__all__ = [
    "CITYSCAPES_CLASSES",
    "IGNORE_INDEX",
    "NightCity",
    "cityscapes_train_ids",
]

import torch
import torch.nn.functional as F
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, register_dataset
from mon.data.structures import ClassDef, ClassList, Sample

IGNORE_INDEX = 255
"""Mask value for pixels outside the evaluated classes (Cityscapes convention)."""

CITYSCAPES_CLASSES = ClassList(
    ClassDef(name=name, id=i, color=color, source_id=label_id)
    for i, (name, label_id, color) in enumerate([
        ("road",          7,  (128, 64, 128)),
        ("sidewalk",      8,  (244, 35, 232)),
        ("building",      11, (70, 70, 70)),
        ("wall",          12, (102, 102, 156)),
        ("fence",         13, (190, 153, 153)),
        ("pole",          17, (153, 153, 153)),
        ("traffic_light", 19, (250, 170, 30)),
        ("traffic_sign",  20, (220, 220, 0)),
        ("vegetation",    21, (107, 142, 35)),
        ("terrain",       22, (152, 251, 152)),
        ("sky",           23, (70, 130, 180)),
        ("person",        24, (220, 20, 60)),
        ("rider",         25, (255, 0, 0)),
        ("car",           26, (0, 0, 142)),
        ("truck",         27, (0, 0, 70)),
        ("bus",           28, (0, 60, 100)),
        ("train",         31, (0, 80, 100)),
        ("motorcycle",    32, (0, 0, 230)),
        ("bicycle",       33, (119, 11, 32)),
    ])
)
"""The 19 evaluated Cityscapes classes: ``id`` = train id, ``source_id`` = ``labelId``, Cityscapes colors."""

_LUT = torch.full((256,), IGNORE_INDEX, dtype=torch.uint8)
for _c in CITYSCAPES_CLASSES:
    _LUT[_c.source_id] = _c.id


def cityscapes_train_ids(label_ids: torch.Tensor) -> torch.Tensor:
    """Map a Cityscapes ``labelId`` map (uint8) to train ids; other ids → ``IGNORE_INDEX``.

    Examples:
        >>> cityscapes_train_ids(torch.tensor([[7, 0, 33, 255]], dtype=torch.uint8)).tolist()
        [[0, 255, 18, 255]]
    """
    return _LUT[label_ids.long()]


@register_dataset("nightcity", tasks={Task.SEGMENT, Task.LLE})
class NightCity(FolderDataset):
    """NightCity: 2,998 train / 1,299 val night street scenes (1024×512), 19 Cityscapes classes.

    ``val`` uses NightLab's re-annotated labels (downsampled to the image size, nearest).
    """

    folder  = "nightcity"
    layout  = {
        s: {"image": Modality(f"{s.value}/images"), "mask": Modality(f"{s.value}/labels", kind="mask")}
        for s in (Split.TRAIN, Split.VAL)
    }
    classes = CITYSCAPES_CLASSES

    def __getitem__(self, index: int) -> Sample:
        transform, self.transform = self.transform, None
        try:
            sample = super().__getitem__(index)
        finally:
            self.transform = transform
        mask, (h, w) = sample["mask"], sample["image"].shape[-2:]
        if mask.shape != (h, w):
            mask = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0].to(torch.uint8)
        sample["mask"] = tv_tensors.Mask(cityscapes_train_ids(mask))
        return self.transform(sample) if self.transform is not None else sample
