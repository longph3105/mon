"""Unpaired low-light test sets: DICM, Fusion, LIME, MEF, NPE, VV (no ground truth).

Six small no-reference benchmarks, stored together under the dataset id ``unpaired_lol`` as
``unpaired_lol/<set>/test/images/`` with upstream file names. Each set is registered on its own
(``dicm``, ``fusion``, ``lime``, ``mef``, ``npe``, ``vv``) because papers report them separately;
samples are ``{"image", "meta"}``. Evaluate with no-reference metrics (e.g. NIQE, BRISQUE).

Warning:
    Papers use different versions of these sets (DICM 44/64/69, NPE 8–85 images): these are DICM
    69, Fusion 16, LIME 10, MEF 17, NPE 8, VV 24. Say which version when comparing with published
    numbers. Four Fusion files are BMP despite their ``.jpg`` suffix; ``read_image`` decodes by
    content. Sizes range from 256×205 to 2304×1769: pad or tile, don't resize, for evaluation.

Examples:
    >>> from mon.core.enums import Split, Task
    >>> from mon.data.dataset import DATASETS
    >>> sorted(UNPAIRED_LOL_SETS) == ["dicm", "fusion", "lime", "mef", "npe", "vv"]
    True
    >>> set(UNPAIRED_LOL_SETS) <= set(DATASETS.names(tasks=Task.LLE, splits=Split.TEST))
    True
    >>> DICM.folder
    'unpaired_lol/dicm'
    >>> ds = DATASETS.build("lime", "test")  # needs DATA_DIR  # doctest: +SKIP
    >>> len(ds), sorted(ds[0])               # doctest: +SKIP
    (10, ['image', 'meta'])

References:
    - DICM: Lee et al., "Contrast Enhancement Based on Layered Difference Representation of 2D
      Histograms", IEEE TIP 2013. http://mcl.korea.ac.kr/projects/LDR/
    - Fusion: Fu et al., "A Fusion-Based Enhancing Method for Weakly Illuminated Images", Signal
      Processing 2016 (images from https://github.com/TwoTu/MF-LIME).
    - LIME: Guo et al., "LIME: Low-Light Image Enhancement via Illumination Map Estimation",
      IEEE TIP 2017.
    - MEF: Ma et al., "Perceptual Quality Assessment for Multi-Exposure Image Fusion", IEEE TIP 2015.
    - NPE: Wang et al., "Naturalness Preserved Enhancement Algorithm for Non-Uniform Illumination
      Images", IEEE TIP 2013.
    - VV: Vonikakis, "Busting image enhancement and tone-mapping algorithms".
      https://sites.google.com/site/vonikakis/datasets
    - Download used (DICM from the official zip): Retinexformer's no-ground-truth sets,
      https://github.com/caiyuanhao1998/Retinexformer
"""

__all__ = [
    "DICM",
    "LIME",
    "MEF",
    "NPE",
    "UNPAIRED_LOL_SETS",
    "VV",
    "Fusion",
]

from mon.core.enums import Task
from mon.data.dataset import FolderDataset, register_dataset, unpaired_layout


@register_dataset("dicm", tasks={Task.LLE})
class DICM(FolderDataset):
    """DICM: 69 photos from commercial digital cameras."""

    folder = "unpaired_lol/dicm"
    layout = unpaired_layout("test")


@register_dataset("fusion", tasks={Task.LLE})
class Fusion(FolderDataset):
    """Fusion (Fu et al. 2016): 16 weakly illuminated images."""

    folder = "unpaired_lol/fusion"
    layout = unpaired_layout("test")


@register_dataset("lime", tasks={Task.LLE})
class LIME(FolderDataset):
    """LIME: 10 low-light images."""

    folder = "unpaired_lol/lime"
    layout = unpaired_layout("test")


@register_dataset("mef", tasks={Task.LLE})
class MEF(FolderDataset):
    """MEF: 17 under-exposed images from multi-exposure sequences."""

    folder = "unpaired_lol/mef"
    layout = unpaired_layout("test")


@register_dataset("npe", tasks={Task.LLE})
class NPE(FolderDataset):
    """NPE: 8 non-uniformly illuminated images."""

    folder = "unpaired_lol/npe"
    layout = unpaired_layout("test")


@register_dataset("vv", tasks={Task.LLE})
class VV(FolderDataset):
    """VV: 24 high-resolution photos with dark and bright regions."""

    folder = "unpaired_lol/vv"
    layout = unpaired_layout("test")


UNPAIRED_LOL_SETS: dict[str, type[FolderDataset]] = {
    "dicm": DICM, "fusion": Fusion, "lime": LIME, "mef": MEF, "npe": NPE, "vv": VV,
}
"""Registry name → class for the six sets, in name order."""
