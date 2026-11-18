"""Global constants: file extensions, dir names, env vars."""

__all__ = [
    "ANNS_DIR",
    "CONFIG_EXTENSIONS",
    "DATASETS_DIR_ENV",
    "DATA_DIR_ENV",
    "DEBUG_DIR",
    "DEFAULT_CKPT_EXT",
    "DEFAULT_IMAGE_EXT",
    "DEFAULT_LABEL_EXT",
    "DEFAULT_WEIGHTS_EXT",
    "DEPTH_DIR",
    "IMAGES_DIR",
    "IMAGE_EXTENSIONS",
    "LABELS_DIR",
    "RAW_IMAGE_EXTENSIONS",
    "RUNS_DIR_ENV",
    "VIDEO_EXTENSIONS",
    "WEIGHTS_EXTENSIONS",
    "ZOO_DIR_ENV",
]

# --- Environment Variables ---

DATA_DIR_ENV    : str = "DATA_DIR"
"""Data root; ``$DATA_DIR/zoo``, ``$DATA_DIR/datasets`` and ``$DATA_DIR/runs`` are the fallbacks."""

DATASETS_DIR_ENV: str = "DATASETS_DIR"
"""Directory holding datasets; see ``mon.core.path.datasets_dir``."""

RUNS_DIR_ENV    : str = "RUNS_DIR"
"""Directory holding run outputs (training, prediction); see ``mon.core.path.runs_dir``."""

ZOO_DIR_ENV     : str = "ZOO_DIR"
"""Directory holding pretrained weights; see ``mon.core.path.zoo_dir``."""

# --- Directory Names ---
# Standard subdirectory names inside datasets and run outputs.

ANNS_DIR  : str = "anns"
DEBUG_DIR : str = "debug"
DEPTH_DIR : str = "depth"
IMAGES_DIR: str = "images"
LABELS_DIR: str = "labels"

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
    ".txt",
    ".yaml",
    ".yml",
})

# --- Default Extensions ---
# Each is a member of the matching ``*_EXTENSIONS`` set.

DEFAULT_CKPT_EXT   : str = ".ckpt"
DEFAULT_IMAGE_EXT  : str = ".jpg"
DEFAULT_LABEL_EXT  : str = ".txt"
DEFAULT_WEIGHTS_EXT: str = ".pt"
