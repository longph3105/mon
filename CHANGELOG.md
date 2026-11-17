# Changelog

## 3.0.0.dev0 — Unreleased

### Added

- `mon.core.path.Path`: `pathlib.Path` subclass with case-insensitive extension checks
  (`has_ext`, `is_image_file`, ...), sorted `files()`/`subdirs()`, `sibling`, `find_sibling`,
  `resolve_subdir`, `ensure_dir`, `normalize`, `size`, `modified_at`, `fingerprint`.
- `mon.core.config`: `load_config` (dataclass schema ← `_base_` files ← YAML ← dotlist overrides, via
  OmegaConf) and `save_config` (resolved, atomic).
- `mon.core.console`: shared `rich` `console`, `get_logger`, `setup_logging` (rich terminal output, optional
  log file, rank-0-only under DDP, captures `warnings`), `print_dict` (also for OmegaConf configs).
  Importing `mon` no longer changes logging levels.
- `mon.core.console.progress_bar`: rich progress bar (speed as `it/s` or `ms/it`), drawn only on rank 0.
- `mon.core.fileio`: `load_json`/`load_yaml`/`load_txt` and atomic `save_json`/`save_yaml`/`save_txt`
  (UTF-8, parent dirs created, paths/NumPy/tensors serialized, `overwrite=False` raises `FileExistsError`).
- `mon.core.enums`: `Task`, `Split`, `RunMode`, `Strategy`, `BBoxFormat` (stdlib `StrEnum` with aliases,
  case-insensitive lookup; unknown values raise). `AliasStrEnum` base for new enums.
- `mon.core.path.zoo_dir()`: weights directory from `MON_ZOO_DIR`, falling back to `$DATA_DIR/zoo`.
- `mon.core.constants`: standard directory names (`IMAGES_DIR`, ...) and default extensions.
- `mon.core.registry.Registry`: case-insensitive name → object registry with metadata filtering
  (`names(**criteria)`), "did you mean" errors, and duplicate protection.
- `mon.core.runtime`: `available_devices()`, `select_device()` (raises instead of silently falling back
  to CPU; `"auto"` = `cuda:0` → `mps` → `cpu`), `seed_everything()` (returns the seed; optional
  deterministic mode).
- `mon.core.weights`: `Weights` (zoo-relative or absolute path, optional URL + SHA-256),
  `WeightsEnum`, `resolve_weights()`; loads with `weights_only=True`; missing files raise.
- `mon.core.size.Size`: validated `(height, width)`; `from_any`, `from_shape(layout=)`, `round_up`.
- `mon.core.timing`: `Timer` and `Profiler` (named stages); sync CUDA and MPS at start and stop.
- `mon.core.type_aliases`: `TensorOrArray`, `Int2`, `Int3`, `Float3`, `OneOrPair[T]`.
- `mon.core.fileio.download_file`: atomic, SHA-256-verified downloads (via `torch.hub`); `sha256sum`.
- `mon.core.path.find_weights`: find a weights file/dir as given, else under `zoo_dir()`.
- `mon.core.utils`: `has_value`, `is_int_like`, `is_list_of`, `to_list` (literal separators, `cast=`),
  `truncate_string` (standard library only).
- `mon.core.constants`: file-extension sets (`IMAGE_EXTENSIONS`, `RAW_IMAGE_EXTENSIONS`, ...).
- Packaging with uv and the `uv_build` backend; install with `uv sync`.
- Minimal runtime dependencies: `numpy`, `torch`, `torchvision`.
- Runtime dependencies: `pyyaml`, `rich`, `tqdm`,
  `xmltodict`, `tensordict`, `omegaconf`,
  `tensorboard`.
- `opencv-python-headless` as a runtime dependency.
- `py.typed` marker (PEP 561): type checkers use `mon`'s annotations.
- Supports Python >= 3.13 (except 3.14.1), with no upper bound.
- `gui` extra (`mon[gui]`): `matplotlib` for displaying images, `PySide6-Essentials` for GUI apps.
