# Changelog

## 3.0.0.dev0 — Unreleased

### Removed

- Unused runtime dependencies `tensordict` (samples are plain dicts) and `tensorboard` (returns with the runners).

### Added

- `mon.models.lle.dccnet`: `DCCNet` (`forward(image) -> {"enhanced", "gray", "color_hist"}`; `H, W` divisible
  by 8), released weights `DCCNet_Weights.LOL_V1` (official Google Drive file, URL, SHA-256; its
  `nn.DataParallel` `module.` prefix is stripped on load). Matches the official network; reproduces the official
  LOL-v1 results (22.976 dB). No training loss (none released).
- `mon.models.lle.clode`: `CLODE` (`forward(image, t_end=None) -> {"enhanced", "curve_map", "noise", "state"}`;
  `t_end` 3 by default, larger brightens, negative darkens), released weights `CLODE_Weights.LOL_V1/SICE_ME/
  UNIVERSAL` (official Drive files without their unused modules; SHA-256; `UNIVERSAL` is the default). Outputs
  match the official code. Outputs depend on the other images in a batch. No training loss yet.
- Runtime dependency `torchdiffeq` (ODE solvers, MIT; CLODE).
- `mon.models.lle.pairlie`: `PairLIE` (`forward(image) -> {"enhanced", "illumination", "reflectance",
  "denoised"}`, `alpha` 0.2 or 0.14 for LOL), `PairLIELoss`, released weights `PairLIE_Weights.SICE` (official
  `weights/PairLIE.pth`, pinned URL, SHA-256). Outputs match the official code bit for bit.
- `mon.models.lle.colie`: CoLIE (`CoLIE` implicit network, `CoLIELoss`, `colie_enhance` fitting one network per
  image with the paper's settings, `fit_illumination`, `fast_guided_filter`, `replace_value`). Registered with
  `zero_shot=True`; the per-image fit matches the official training.
- `mon.models.lle.sci`: `SCI` and `SCIPP` (`forward(image) -> {"enhanced", "illumination"}`, `forward_stages` for
  training), `SCILoss`, released weights (`SCI_Weights.EASY/MEDIUM/DIFFICULT`, `SCIPP_Weights.TPAMI`; pinned URLs,
  SHA-256). Released checkpoints load strictly; outputs match the official code bit for bit.
- `mon.models.denoise.tv_denoise`: total-variation denoising (`tv_denoise`, parameter-free `TVDenoise`), same
  iteration as scikit-image's `denoise_tv_chambolle`; batched, differentiable, any device.
- `mon.models.denoise.zs_n2n`: Zero-Shot Noise2Noise (`ZSN2N` network, `ZSN2NLoss`, `pair_downsample`,
  `zs_n2n_denoise` fitting a fresh network per image, seedable). Registered with `zero_shot=True`.
- Dev dependency `scikit-image` (test reference for TV denoising).
- `mon.models.mef.mertens`: Mertens exposure fusion (`Mertens` module, `exposure_fusion`, `fusion_weights`,
  `pyr_down`/`pyr_up`), registered as `mertens` (`Task.MEF`). Input `[B, N, C, H, W]` exposure stacks; GPU and
  autograd. Matches OpenCV's `MergeMertens` and fixes `mon_old`'s version (crashed on every input; well-exposedness
  5× too wide; fixed 4 pyramid levels; grayscale stacks averaged).
- `mon.models.load`: `weights="default"` loads nothing for parameter-free models.
- `mon.models` package: `MODELS` registry listing every model lazily (filtering imports no model code or
  PyTorch) and `load(name, weights="default", device=, strict=, **kwargs)`, which returns the model in eval mode.
  `mon.models.lle.zero_dce`: `ZeroDCE`, `ZeroDCEPP` (`forward(image) -> {"enhanced", "curve"}`), the released
  weights (`ZeroDCE_Weights`/`ZeroDCEPP_Weights.SICE_ME`, pinned URLs and SHA-256), `ZeroDCELoss`, and
  `apply_curves`. Outputs match the released models bit for bit; Zero-DCE++ accepts any input size with
  `scale_factor > 1`.
- `Registry.add_lazy(name, "module:Attr", **meta)`: entries imported on first lookup; `names()` never imports.
  `Entry.target` records the import path.
- `mon.nn.losses.reduction`: `Reduction`, `check_reduction`, `reduce_loss`, shared by all losses.
- `.markdownlint.yaml` (120 columns; tables and code blocks exempt).
- `mon.data.dataset.ClassFolderDataset` + `classification_layout`: classification from `<split>/<class>/<image>`
  with label ids from `classes.txt`; samples `{"image", "label", "meta"}`. `mon.data.zoo.classify`: `cifar10`,
  `cifar100`, `mnist`, `fashion_mnist` (`Task.CLASSIFY`).
- `mon.nn` package (lazy namespace). `mon.nn.losses` (one module per family). `mon.nn.losses.zero_reference`:
  Zero-DCE's zero-reference losses (`ColorConstancyLoss`, `ExposureControlLoss`, `SpatialConsistencyLoss`,
  `IlluminationSmoothnessLoss`), matching the official code, with `reduction="mean"|"sum"|"none"`.
  `mon.nn.tiling.tiled_forward`: Hann-blended overlapping-tile inference (full coverage, borders kept).
  `mon.nn.utils`: `count_parameters`, `freeze`/`unfreeze`, `module_device`.
- `mon.data.dataset.ImageFolder`: prediction input from a folder (recursive, sorted), one file, or a list of
  files; `meta.stem` keeps the relative path so outputs can mirror the input tree. Folder indexing (also in
  `FolderDataset`) skips hidden files such as macOS `._*` AppleDouble files.
- `docs/data.md`: guide to datasets, transforms, loaders, and pad-and-crop inference; its code runs in the
  tests. Docstring examples for `register_dataset`/`DATASETS` and the LOL datasets.
- `Registry.add`: an object with the same `__module__` and `__qualname__` as the registered one (its module
  imported again: `importlib.reload`, notebooks, pytest) replaces it instead of raising.
- `mon.data.zoo.lol`: `LOLv1` (`lol_v1`) and `LOLv2Real` (`lol_v2_real`), registered in `DATASETS` for
  `Task.LLE`. `mon.data.dataset.paired_layout` for the normalized `<split>/images|targets` layout.
- `mon.data.transforms`: `ApplyTo` (run a transform on selected sample keys, shared random parameters),
  `ResizeToMultiple` (training), `PadToMultiple`/`pad_to_multiple` + `crop_to` (exact pad-and-crop inference),
  `MinMaxNormalize` (per-image or per-channel `[0, 1]` scaling).
- `python -m mon.core.path`: shows the data, datasets, and zoo directories, their source variable, and
  whether they exist. `datasets_dir()`/`zoo_dir()` log the resolved directory and source once.
- Environment variables are `DATA_DIR`, `DATASETS_DIR`, `ZOO_DIR` (no `MON_` prefix).
- `mon.data.dataset`: `FolderDataset` (declare per-split modality folders; pairing by relative stem),
  `Modality` (image/mask/depth/yolo/voc), `DATASETS` registry (zoo loaded on first lookup),
  `register_dataset`. `mon.data.loader`: `collate`, `create_dataloader` (seeded shuffle and workers).
- `mon.core.path.datasets_dir()` (`DATASETS_DIR`, else `$DATA_DIR/datasets`); `Registry(autoload=)`.
- `mon.data.structures`: `ClassDef`, `ClassList` (validated, name/id lookup, `encode` annotation ids → training
  ids, `from_coco` with raw ids kept as `source_id`, YAML I/O), `SampleMeta`, `Sample`.
- `mon.data.labels`: `read_yolo`/`write_yolo`, `read_voc`, `read_coco` (`CocoAnnotations`) → sample dicts
  with `tv_tensors.BoundingBoxes` (absolute XYXY) and `labels`; malformed files raise with file and line.
- `mon.data` package (lazy namespace). `mon.data.io`: `read_image`/`write_image` (RGB `tv_tensors.Image`,
  `uint8`/`uint16`, EXIF orientation, atomic writes; zero-copy CHW view), `to_numpy_hwc`, `iter_video_frames`,
  `video_info`, `VideoWriter` (OpenCV).
- Flat, lazily loaded namespace: `import mon.core as mc; mc.Path, mc.load_config, …`;
  `mon.core` subpackage reachable from `import mon`. `get_console()` accessor.
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
- `mon.core.path.zoo_dir()`: weights directory from `ZOO_DIR`, falling back to `$DATA_DIR/zoo`.
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
- Runtime dependencies: `pyyaml`, `rich`, `tqdm`, `omegaconf`.
- `opencv-python-headless` as a runtime dependency.
- `py.typed` marker (PEP 561): type checkers use `mon`'s annotations.
- Supports Python >= 3.13 (except 3.14.1), with no upper bound.
- `gui` extra (`mon[gui]`): `matplotlib` for displaying images, `PySide6-Essentials` for GUI apps.
