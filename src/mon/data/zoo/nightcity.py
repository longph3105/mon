"""NightCity: night-time semantic segmentation (with NightLab's re-annotated val), Cityscapes classes.

Also tagged ``Task.LLE``: night images with no reference, used to evaluate enhancement
(no-reference metrics) and enhancement-for-segmentation.

Stored in the normalized segmentation layout (workspace ``data/README.md``): ``<split>/images/``
and single-channel PNG class maps in ``<split>/labels/``, with Cityscapes ``labelId`` values.
Samples are ``{"image", "mask", "meta"}``; ``mask`` holds Cityscapes **train ids** (0–18, the
19 evaluated classes of ``classes``; every other ``labelId`` and 255 → ``IGNORE_INDEX`` 255; label
definitions in ``mon.data.zoo.cityscapes``).

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
    "NightCity",
]

import torch
import torch.nn.functional as F
from torchvision import tv_tensors

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, register_dataset
from mon.data.structures import Sample
from mon.data.zoo.cityscapes import CITYSCAPES_CLASSES, cityscapes_train_ids


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

    def postprocess(self, sample: Sample) -> Sample:
        """Downsample a larger mask to the image size (nearest), then map ``labelId``s to train ids."""
        mask, (h, w) = sample["mask"], sample["image"].shape[-2:]
        if mask.shape != (h, w):
            mask = F.interpolate(mask[None, None].float(), size=(h, w), mode="nearest")[0, 0].to(torch.uint8)
        sample["mask"] = tv_tensors.Mask(cityscapes_train_ids(mask))
        return sample
