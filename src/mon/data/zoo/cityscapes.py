"""Cityscapes label definitions, shared by datasets annotated with Cityscapes ``labelId``s (NightCity).

``CITYSCAPES_CLASSES`` lists the 19 evaluated classes (``id`` = train id 0–18, ``source_id`` =
``labelId``, Cityscapes colors); ``cityscapes_train_ids`` maps a ``labelId`` map to train ids, every
other id becoming ``IGNORE_INDEX`` (255). The Cityscapes dataset itself would also live here.

References:
    - Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding", CVPR 2016.
      https://github.com/mcordts/cityscapesScripts (``helpers/labels.py``)
"""

__all__ = [
    "CITYSCAPES_CLASSES",
    "IGNORE_INDEX",
    "cityscapes_train_ids",
]

import torch

from mon.data.structures import ClassDef, ClassList

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
