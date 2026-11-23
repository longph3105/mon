"""Global constants: file extensions, env vars."""

__all__ = [
    "CONFIG_EXTENSIONS",
    "DATASETS_DIR_ENV",
    "DATA_DIR_ENV",
    "IMAGE_EXTENSIONS",
    "RAW_IMAGE_EXTENSIONS",
    "RUNS_DIR_ENV",
    "VIDEO_EXTENSIONS",
    "WEIGHTS_EXTENSIONS",
    "ZOO_DIR_ENV",
]

from mon.core._lazy import deprecated_getattr

# --- Environment Variables ---

DATA_DIR_ENV    : str = "DATA_DIR"
"""Data root; ``$DATA_DIR/zoo``, ``$DATA_DIR/datasets`` and ``$DATA_DIR/runs`` are the fallbacks."""

DATASETS_DIR_ENV: str = "DATASETS_DIR"
"""Directory holding datasets; see ``mon.core.path.datasets_dir``."""

RUNS_DIR_ENV    : str = "RUNS_DIR"
"""Directory holding run outputs (training, prediction); see ``mon.core.path.runs_dir``."""

ZOO_DIR_ENV     : str = "ZOO_DIR"
"""Directory holding pretrained weights; see ``mon.core.path.zoo_dir``."""

# --- File Extensions ---
# Lowercase, with a leading dot. Compare against ``Path.suffix.lower()``.

RAW_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".arw",   # Sony
    ".cr2",   # Canon
    ".dng",   # Adobe Digital Negative
    ".nef",   # Nikon
    ".raf",   # Fujifilm
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".bmp",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}) | RAW_IMAGE_EXTENSIONS

VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
})

WEIGHTS_EXTENSIONS: frozenset[str] = frozenset({
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".weights",
})

CONFIG_EXTENSIONS: frozenset[str] = frozenset({
    ".cfg",
    ".config",
    ".json",
    ".yaml",
    ".yml",
})

# --- Deprecated ---
# Unused in mon; removed in 3.1. Access still works, with a ``DeprecationWarning``.

__getattr__ = deprecated_getattr(__name__, {
    "ANNS_DIR"           : "anns",
    "DEBUG_DIR"          : "debug",
    "DEPTH_DIR"          : "depth",
    "IMAGES_DIR"         : "images",
    "LABELS_DIR"         : "labels",
    "DEFAULT_CKPT_EXT"   : ".ckpt",
    "DEFAULT_IMAGE_EXT"  : ".jpg",
    "DEFAULT_LABEL_EXT"  : ".txt",
    "DEFAULT_WEIGHTS_EXT": ".pt",
}, removal="3.1")
