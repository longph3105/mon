"""SICE multi-exposure datasets: SICE (Part 1 / Part 2) and SICE-ME (the Zero-DCE training split).

SICE holds multi-exposure sequences of static scenes (3–28 exposures each) with one reference
image per scene, as 8-bit sRGB JPEG (some test targets PNG) of 2448×3264 to 5472×3648 pixels.
``sice`` is stored in the multi-exposure layout (``<split>/images/<scene>/<k>.<ext>``,
``<split>/targets/<scene>.<ext>``); ``sice_me`` is derived from it in the paired layout
(``<split>/images/<scene>_<k>.<ext>``). Images are full resolution: resize in the transform.

Warning:
    ``sice_me`` is built from ``sice`` train (Part 1) only: never evaluate a model trained on
    ``sice_me`` on ``sice`` train. Its train/val split is per image, as in Zero-DCE, so 288 of
    the 360 scenes have exposures in both: ``sice_me`` val scores are optimistic. The upstream
    ``test/testing_index.txt`` defines a test subset from Part 1 scenes, i.e. ``sice`` train.
    Exposures of one scene share a target, so never re-split ``sice`` randomly by image for
    evaluation. 12 exposure pairs are byte-identical upstream (test scenes 109–119, ``5`` = ``6``;
    train scene 140, ``3`` = ``4``).

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> {"sice", "sice_me"} <= set(DATASETS.names(tasks=Task.LLE))
    True
    >>> SICE.splits() == {Split.TRAIN, Split.TEST}, SICEME.splits() == {Split.TRAIN, Split.VAL}
    (True, True)
    >>> SICE.pair_key("target", "12/3")
    '12'
    >>> ds = DATASETS.build("sice", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), ds[0]["meta"].stem          # doctest: +SKIP
    (1779, '1/1')

References:
    - Cai et al., "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure Images",
      IEEE TIP 2018. https://github.com/csjcai/SICE
    - SICE-ME split: Guo et al., "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement", CVPR 2020, Sec. 4 (https://arxiv.org/abs/2001.06826). Built by the
      workspace ``scripts/make_sice_me.sh`` (seed 1143, 2,422 train / 599 val).
"""

__all__ = [
    "SICE",
    "SICEME",
]

from mon.core.enums import Split, Task
from mon.data.dataset import FolderDataset, Modality, paired_layout, register_dataset


@register_dataset("sice", tasks={Task.LLE})
class SICE(FolderDataset):
    """SICE: 360 train (Part 1, 3,021 exposures) and 229 test (Part 2, 1,779 exposures) scenes.

    One sample per exposure, ``{"image", "target", "meta"}``, with ``target`` the scene's
    reference image and ``meta.stem`` = ``"<scene>/<k>"``.
    """

    folder = "sice"
    layout = paired_layout("train", "test")

    @classmethod
    def pair_key(cls, modality: str, key: str) -> str:
        """Exposure ``"<scene>/<k>"`` → target ``"<scene>"``."""
        return key.split("/", 1)[0] if modality == "target" else key


@register_dataset("sice_me", tasks={Task.LLE})
class SICEME(FolderDataset):
    """SICE-ME: Zero-DCE's split of SICE Part 1, 2,422 train images (no targets) and 599 val pairs.

    ``meta.stem`` is ``"<scene>_<k>"``.
    """

    folder = "sice_me"
    layout = {
        Split.TRAIN: {"image": Modality("train/images")},
        **paired_layout("val"),
    }
