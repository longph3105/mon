"""Data handling: files on disk → samples → batches.

Covers image/video/label I/O, sample structures (``tv_tensors``), datasets, transforms,
and data loaders. Image-processing algorithms live in ``mon.ops``, drawing in
``mon.plot``, metrics in ``mon.metrics``. Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    ==========  ===================================================================  ==========================================
    Module      Purpose                                                              Key names
    ==========  ===================================================================  ==========================================
    dataset     Folder-based datasets declared by their on-disk layout.              FolderDataset, ImageFolder, DATASETS
    io          Read and write images and videos.                                    read_image, write_image, VideoWriter
    labels      Read and write YOLO, VOC, and COCO box labels.                       read_yolo, read_voc, read_coco, write_yolo
    loader      Batching for dict samples and reproducible DataLoader construction.  collate, create_dataloader
    structures  Class definitions and per-sample metadata.                           ClassList, ClassDef, SampleMeta
    transforms  Sample transforms missing from torchvision v2.                       ApplyTo, PadToMultiple, ResizeToMultiple
    ==========  ===================================================================  ==========================================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("dataset", "io", "labels", "loader", "structures", "transforms")

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # dataset
    "DATASETS"             : "dataset",
    "ClassFolderDataset"   : "dataset",
    "FolderDataset"        : "dataset",
    "ImageFolder"          : "dataset",
    "Modality"             : "dataset",
    "ModalityKind"         : "dataset",
    "classification_layout": "dataset",
    "detection_layout"     : "dataset",
    "paired_layout"        : "dataset",
    "register_dataset"     : "dataset",
    "unpaired_layout"      : "dataset",
    # io
    "VideoInfo"            : "io",
    "VideoWriter"          : "io",
    "iter_video_frames"    : "io",
    "read_image"           : "io",
    "to_numpy_hwc"         : "io",
    "video_info"           : "io",
    "write_image"          : "io",
    # labels
    "CocoAnnotations"      : "labels",
    "read_coco"            : "labels",
    "read_voc"             : "labels",
    "read_yolo"            : "labels",
    "write_yolo"           : "labels",
    # loader
    "LIST_KEYS"            : "loader",
    "collate"              : "loader",
    "create_dataloader"    : "loader",
    # structures
    "ClassDef"             : "structures",
    "ClassList"            : "structures",
    "Sample"               : "structures",
    "SampleMeta"           : "structures",
    # transforms
    "ApplyTo"              : "transforms",
    "MinMaxNormalize"      : "transforms",
    "PadToMultiple"        : "transforms",
    "ResizeToMultiple"     : "transforms",
    "crop_to"              : "transforms",
    "pad_to_multiple"      : "transforms",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "ApplyTo",
    "ClassDef",
    "ClassFolderDataset",
    "ClassList",
    "CocoAnnotations",
    "DATASETS",
    "FolderDataset",
    "ImageFolder",
    "LIST_KEYS",
    "MinMaxNormalize",
    "Modality",
    "ModalityKind",
    "PadToMultiple",
    "ResizeToMultiple",
    "Sample",
    "SampleMeta",
    "VideoInfo",
    "VideoWriter",
    "classification_layout",
    "collate",
    "create_dataloader",
    "crop_to",
    "detection_layout",
    "iter_video_frames",
    "pad_to_multiple",
    "paired_layout",
    "read_coco",
    "read_image",
    "read_voc",
    "read_yolo",
    "register_dataset",
    "to_numpy_hwc",
    "unpaired_layout",
    "video_info",
    "write_image",
    "write_yolo",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.data.dataset import (
        DATASETS, ClassFolderDataset, FolderDataset, ImageFolder, Modality, ModalityKind, classification_layout,
        detection_layout, paired_layout, register_dataset, unpaired_layout,
    )
    from mon.data.io import (
        VideoInfo, VideoWriter, iter_video_frames, read_image, to_numpy_hwc, video_info, write_image,
    )
    from mon.data.labels import CocoAnnotations, read_coco, read_voc, read_yolo, write_yolo
    from mon.data.loader import LIST_KEYS, collate, create_dataloader
    from mon.data.structures import ClassDef, ClassList, Sample, SampleMeta
    from mon.data.transforms import (
        ApplyTo, MinMaxNormalize, PadToMultiple, ResizeToMultiple, crop_to, pad_to_multiple,
    )
