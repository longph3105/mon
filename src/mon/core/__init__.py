"""Core functionality shared by the rest of ``mon``.

Public names are re-exported lazily (PEP 562): ``mon.core.Path`` imports only
``mon.core.path``, and torch loads only when a name that needs it is first used.
Code inside ``mon`` imports from the submodules directly (``from mon.core.path import Path``).

Submodules (kept in sync with the code by ``tests/core/test_init.py``):

    ============  =======================================================  =========================================
    Module        Purpose                                                  Key names
    ============  =======================================================  =========================================
    config        Load and save OmegaConf configs.                         load_config, save_config
    console       Logging, progress bars, and the shared rich console.     get_logger, setup_logging, progress_bar
    constants     Global constants: file extensions, dir names, env vars.  IMAGE_EXTENSIONS, IMAGES_DIR, ZOO_DIR_ENV
    enums         Shared string enums with aliases.                        Task, Split, RunMode, BBoxFormat
    fileio        Atomic JSON/YAML/text I/O and verified downloads.        load_yaml, save_json, download_file
    path          Path subclass plus model-zoo and weights lookup.         Path, zoo_dir, find_weights
    registry      Name-to-object registries with metadata filtering.       Registry
    runtime       Device selection and global RNG seeding.                 select_device, seed_everything
    size          Unambiguous (height, width) image size.                  Size
    timing        Wall-clock timers that account for async GPU work.       Timer, Profiler
    type_aliases  Shared type aliases for annotations.                     TensorOrArray, Int2, OneOrPair
    utils         Stdlib-only helpers for values, strings, and lists.      has_value, to_list, truncate_string
    weights       Pretrained-weights descriptors, enums, and loading.      Weights, WeightsEnum, resolve_weights
    ============  =======================================================  =========================================

Examples:
    >>> import mon.core as mc
    >>> mc.Size(224, 224).hw
    (224, 224)
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = (
    "config", "console", "constants", "enums", "fileio", "path", "registry",
    "runtime", "size", "timing", "type_aliases", "utils", "weights",
)

# Public name -> submodule. Keep in sync with each submodule's ``__all__`` (enforced by
# ``tests/core/test_init.py``). ``console.console`` is omitted: it collides with the
# ``console`` submodule name; use ``get_console()``.
_EXPORTS: dict[str, str] = {
    # config
    "BASE_KEY"             : "config",
    "load_config"          : "config",
    "save_config"          : "config",
    # console
    "DEFAULT_QUIET_LOGGERS": "console",
    "SpeedColumn"          : "console",
    "get_console"          : "console",
    "get_logger"           : "console",
    "print_dict"           : "console",
    "progress_bar"         : "console",
    "setup_logging"        : "console",
    # constants
    "ANNS_DIR"             : "constants",
    "CONFIG_EXTENSIONS"    : "constants",
    "DATA_DIR_ENV"         : "constants",
    "DEBUG_DIR"            : "constants",
    "DEFAULT_CKPT_EXT"     : "constants",
    "DEFAULT_IMAGE_EXT"    : "constants",
    "DEFAULT_LABEL_EXT"    : "constants",
    "DEFAULT_WEIGHTS_EXT"  : "constants",
    "DEPTH_DIR"            : "constants",
    "IMAGES_DIR"           : "constants",
    "IMAGE_EXTENSIONS"     : "constants",
    "LABELS_DIR"           : "constants",
    "RAW_IMAGE_EXTENSIONS" : "constants",
    "VIDEO_EXTENSIONS"     : "constants",
    "WEIGHTS_EXTENSIONS"   : "constants",
    "ZOO_DIR_ENV"          : "constants",
    # enums
    "AliasStrEnum"         : "enums",
    "BBoxFormat"           : "enums",
    "RunMode"              : "enums",
    "Split"                : "enums",
    "Strategy"             : "enums",
    "Task"                 : "enums",
    # fileio
    "download_file"        : "fileio",
    "load_json"            : "fileio",
    "load_txt"             : "fileio",
    "load_yaml"            : "fileio",
    "save_json"            : "fileio",
    "save_txt"             : "fileio",
    "save_yaml"            : "fileio",
    "sha256sum"            : "fileio",
    # path
    "Path"                 : "path",
    "find_weights"         : "path",
    "zoo_dir"              : "path",
    # registry
    "Entry"                : "registry",
    "Registry"             : "registry",
    "RegistryKeyError"     : "registry",
    # runtime
    "available_devices"    : "runtime",
    "seed_everything"      : "runtime",
    "select_device"        : "runtime",
    # size
    "Size"                 : "size",
    # timing
    "Profiler"             : "timing",
    "Timer"                : "timing",
    "synchronize"          : "timing",
    # type_aliases
    "Float3"               : "type_aliases",
    "Int2"                 : "type_aliases",
    "Int3"                 : "type_aliases",
    "OneOrPair"            : "type_aliases",
    "TensorOrArray"        : "type_aliases",
    # utils
    "has_value"            : "utils",
    "is_int_like"          : "utils",
    "is_list_of"           : "utils",
    "to_list"              : "utils",
    "truncate_string"      : "utils",
    # weights
    "Weights"              : "weights",
    "WeightsEnum"          : "weights",
    "WeightsLike"          : "weights",
    "resolve_weights"      : "weights",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "ANNS_DIR",
    "AliasStrEnum",
    "BASE_KEY",
    "BBoxFormat",
    "CONFIG_EXTENSIONS",
    "DATA_DIR_ENV",
    "DEBUG_DIR",
    "DEFAULT_CKPT_EXT",
    "DEFAULT_IMAGE_EXT",
    "DEFAULT_LABEL_EXT",
    "DEFAULT_QUIET_LOGGERS",
    "DEFAULT_WEIGHTS_EXT",
    "DEPTH_DIR",
    "Entry",
    "Float3",
    "IMAGES_DIR",
    "IMAGE_EXTENSIONS",
    "Int2",
    "Int3",
    "LABELS_DIR",
    "OneOrPair",
    "Path",
    "Profiler",
    "RAW_IMAGE_EXTENSIONS",
    "Registry",
    "RegistryKeyError",
    "RunMode",
    "Size",
    "SpeedColumn",
    "Split",
    "Strategy",
    "Task",
    "TensorOrArray",
    "Timer",
    "VIDEO_EXTENSIONS",
    "WEIGHTS_EXTENSIONS",
    "Weights",
    "WeightsEnum",
    "WeightsLike",
    "ZOO_DIR_ENV",
    "available_devices",
    "download_file",
    "find_weights",
    "get_console",
    "get_logger",
    "has_value",
    "is_int_like",
    "is_list_of",
    "load_config",
    "load_json",
    "load_txt",
    "load_yaml",
    "print_dict",
    "progress_bar",
    "resolve_weights",
    "save_config",
    "save_json",
    "save_txt",
    "save_yaml",
    "seed_everything",
    "select_device",
    "setup_logging",
    "sha256sum",
    "synchronize",
    "to_list",
    "truncate_string",
    "zoo_dir",
]


__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.core.config import BASE_KEY, load_config, save_config
    from mon.core.console import (
        DEFAULT_QUIET_LOGGERS, SpeedColumn, get_console, get_logger, print_dict, progress_bar, setup_logging,
    )
    from mon.core.constants import (
        ANNS_DIR, CONFIG_EXTENSIONS, DATA_DIR_ENV, DEBUG_DIR, DEFAULT_CKPT_EXT, DEFAULT_IMAGE_EXT,
        DEFAULT_LABEL_EXT, DEFAULT_WEIGHTS_EXT, DEPTH_DIR, IMAGE_EXTENSIONS, IMAGES_DIR, LABELS_DIR,
        RAW_IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, WEIGHTS_EXTENSIONS, ZOO_DIR_ENV,
    )
    from mon.core.enums import AliasStrEnum, BBoxFormat, RunMode, Split, Strategy, Task
    from mon.core.fileio import (
        download_file, load_json, load_txt, load_yaml, save_json, save_txt, save_yaml, sha256sum,
    )
    from mon.core.path import Path, find_weights, zoo_dir
    from mon.core.registry import Entry, Registry, RegistryKeyError
    from mon.core.runtime import available_devices, seed_everything, select_device
    from mon.core.size import Size
    from mon.core.timing import Profiler, Timer, synchronize
    from mon.core.type_aliases import Float3, Int2, Int3, OneOrPair, TensorOrArray
    from mon.core.utils import has_value, is_int_like, is_list_of, to_list, truncate_string
    from mon.core.weights import Weights, WeightsEnum, WeightsLike, resolve_weights
