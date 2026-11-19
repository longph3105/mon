# Changelog

## 3.0.0.dev0 — Unreleased

### Added

- Data zoo: `mit_5k` (MIT-Adobe FiveK, `mon.data.zoo.fivek`): input → Expert C pairs, 3,500 train / 500 val /
  1,000 test (yuukicammy split), 768 px.
- Data zoo: `mot16` (MOT16, `mon.data.zoo.track`), tagged tracking and detection: per-frame samples, train with
  the pedestrian boxes MOT evaluation counts (YOLO view, clipped to the frame), test frames without labels.
- Data zoo: `uhd_ll` (UHD-LL, UHDFour ICLR 2023): 2,000 train / 150 test 4K pairs from the official release.
- Data zoo: `ntire_2025_llie` (NTIRE 2025 Low Light Image Enhancement challenge): 219 train pairs and the 46 val / 30
  test inputs (no released ground truth), PNG, challenge ids.
- Data zoo: `ydld` (YDLD, NeurIPS 2024: night driving light sources, 3 classes, 1,588 train / 1,928 test), a detection
  dataset also tagged LLE, like `exdark`.
- Data zoo: `loli_street` (LoLI-Street, ACCV 2024): 30,000 train / 3,000 val synthetic low-light images at three
  darkness levels per scene (`<n>/<level>`, paired with the scene's target) and 1,000 real test images (no targets).
- Data zoo: `exdark` (ExDark, official split, 12 classes) and `darkface` (DARK FACE, seeded 5000/1000 hold-out, `face`)
  as detection datasets, also tagged LLE (`{"image", "boxes", "labels"}` from YOLO labels; `detection_layout`), and
  `nightcity` (NightCity with NightLab's re-annotated val) as a segmentation dataset, also tagged LLE (`{"image",
  "mask"}`, Cityscapes train ids, 255 = ignore; `CITYSCAPES_CLASSES`, `cityscapes_train_ids`).
- `FolderDataset.postprocess(sample)`: a hook to fix a sample as read from disk, before `transform` (`NightCity` uses it
  for its masks instead of swapping out `self.transform` in `__getitem__`).
- `docs/efficiency.md`: params, MACs, latency and peak memory of every LLE model on `testkit_efficiency` (RTX 3090),
  plus per-image fit time for the zero-shot CoLIE and Zero-IG.
- `mon benchmark data=testkit_efficiency` (or any dataset id or folder): params, MACs, latency, and peak memory per
  image resolution on real images, with out-of-memory sizes recorded; `sizes=[...]` filters, `output=` writes YAML
  and CSV. New `testkit_efficiency`: 5 dark SICE scenes and their references at 256×256, 512×512, 600×400, 720p,
  1080p, 2K, 4K and 8K (`python -m mon.data.zoo testkit build --part efficiency`; `--part` rebuilds one part).
- No-reference metrics `brisque` (lower is better), `nrqm` (Ma et al., higher is better) and `pi` (PIRM perceptual
  index, lower is better), matching pyiqa 0.1.16 (BRISQUE within 0.1, NRQM and PI within 2e-3 on LOL-v1). NRQM's
  official forests need a one-time conversion: `uv run --with scipy python -m mon.metrics.nrqm convert model.mat`.
  `NIQE(color_space="ycbcr")` and `rgb_to_y` for the YCbCr Y channel (used by PI).
- `lle.sgz` (SGZ, WACV 2022): the Zero-DCE++ network with SGZ's released weights (`sice_me`; the official test uses
  `model_args.scale_factor=12`), checked against the official forward pass.
- `lle.ruas` (RUAS, CVPR 2021) with the three released checkpoints (`lol_v1`, `upe` default, `darkface`); matches the
  official forward pass exactly. `mon predict` saves each checkpoint's official output (`Weights.meta["output_key"]`:
  denoised for `lol_v1`, pre-denoise for `upe`/`darkface`) unless `output_key` is set.
- `mon benchmark model=<name> [size=[H,W]] …` (`mon.runners.benchmarker`) and `mon.nn.complexity`: parameters,
  FLOPs/MACs (`torch.utils.flop_counter`, no new dependency; matmuls and convolutions only), forward latency (mean,
  median, std), and peak CUDA memory, on a seeded random input; `output=` writes the result as YAML.
- `mon.data.zoo.testkit`: `testkit_lle`, `testkit_lle_unpaired`, `testkit_mef`, `testkit_classify`: 100 samples each
  (256×256) for fast tests, built deterministically by `python -m mon.data.zoo testkit build` (manifest with
  source SHA-256s). `ClassFolderDataset.items()`.
- `mon.models.flush_denormals`, applied by `load`: zeroes denormal weights (about 25% of the released Zero-DCE
  weights), which made CPU convolutions ~50× slower; outputs unchanged.
- Tests: `test_testkit_regression.py` (12 LLE models with released weights, ~40 s on GPU); `regression_device`
  fixture (CUDA without TF32, else CPU); `full` marker, off by default, for whole-dataset regressions (now on GPU:
  80 s instead of ~12 min on CPU).
- `mon.runners.trainer`: `TrainConfig`, `resolve_train_config`, `train(cfg)` and `mon train`: one loop for every
  model with a training recipe (AMP, gradient clipping, Adam/AdamW/SGD, step/multistep/cosine schedules,
  validation with the evaluation protocol, `best.pt`/`epoch_NNNN.pt`/`last.ckpt`, exact resume, `log.csv`).
  `cache=true` keeps the transformed training set in RAM or, with `cache_device=device`, in GPU memory
  (identical tensors; deterministic transforms only); `batch_size=auto` probes the largest power-of-two batch under
  85% of GPU memory (opt-in; recipes keep their official batch size).
- `mon.nn.training.TrainRecipe` and recipes `ZERO_DCE_RECIPE`, `ZERO_DCEPP_RECIPE`, `SCI_RECIPE`, `ZERO_IG_RECIPE`
  (official optimizer, epochs, batch, image size, clipping, init), named by the models' `MODELS` entries (`train=`).
  The Zero-DCE loop matches the official `lowlight_train.py` to float32 rounding (parity test).
- `mon.core.utils.ReadOnlyDict`: a read-only mapping that pickles.
- `mon.metrics`: `PSNR`/`psnr`, `SSIM`/`ssim` (`y_channel`, `crop_border`, `downsample`), `LPIPS` (AlexNet or
  VGG-16; official v0.1 layers as `LPIPS_Weights`, pinned URL and SHA-256), `NIQE` (port of BasicSR's, pristine
  model as `NIQE_Weights`), and the lazy `METRICS` index (`mode`, `higher_better`). Every metric returns one
  float64 score per image. Defaults match pyiqa (parity tests on synthetic and LOL-v1 images; LPIPS exact).
- `mon.runners.evaluator`: `EvaluateConfig` and `evaluate(cfg)`, scoring a folder of saved predictions against a
  dataset's targets or a target folder (by stem), or with no-reference metrics alone; optional GT-mean
  protocol; writes `metrics.csv` and `metrics.yaml`. CLI: `mon evaluate [config.yaml] key=value …`.
- `FolderDataset.items()` and `ImageFolder.items()`: each sample's stem and files, without reading them.
- `mon.runners.predictor`: `PredictConfig` and `predict(cfg)`: run a model on a folder, file, list of files, or
  dataset split; pad to the model's `input_multiple` and crop back; fit zero-shot models per image (registry
  `fit`); optional tiling; save the main output (or all image-shaped ones) clamped to `[0, 1]`, mirroring the
  input tree, plus `config.yaml` (with the seed) and `timing.yaml`. CLI: `mon predict [config.yaml] key=value …`
  (`[project.scripts]`).
- `mon.core.path.runs_dir()` (`$RUNS_DIR`, else `$DATA_DIR/runs`) and `RUNS_DIR_ENV`.
- `input_multiple` class attribute on `DCCNet`, `HVICIDNet` (8) and `ZeroIG` (2); registry metadata `fit` on the
  zero-shot models (`zs_n2n`, `colie`, `zero_ig`).
- `mon.models.lle.zero_ig`: Zero-IG (`ZeroIG` network, `ZeroIGLoss(model, image)`, `zero_ig_enhance` fitting a
  fresh network per image with the official recipe), registered `zero_shot=True`, plus the official dataset-trained
  weights `ZeroIG_Weights.LOL_V1` (default) / `LSRW_HUAWEI` / `LSRW_NIKON` (pinned URLs, SHA-256). Inference and the
  loss value match the official code exactly; gradients to float32 rounding.
- `mon.data.zoo.unpaired_lol`: the no-reference test sets `DICM` (`dicm`, 69), `Fusion` (`fusion`, 16), `LIME`
  (`lime`, 10), `MEF` (`mef`, 17), `NPE` (`npe`, 8) and `VV` (`vv`, 24), test split only, `{"image", "meta"}`
  samples; stored under `unpaired_lol/<set>/`. `UNPAIRED_LOL_SETS` maps the names to the classes.
- `mon.data.dataset.unpaired_layout(*splits)`: `<split>/images/` → `image`, for test sets without targets.
- `mon.data.zoo.sice`: `SICE` (`sice`; one sample per exposure, paired with its scene's target; 3,021 train /
  1,779 test) and `SICEME` (`sice_me`, Zero-DCE's per-image split of SICE Part 1: 2,422 train images without
  targets, 599 val pairs), registered for `Task.LLE`.
- `FolderDataset.pair_key(modality, key)`: overridable mapping from a primary file's key to another modality's
  key (default: the same key), for datasets whose targets aren't stored under the input's stem.
- `mon.models.lle.retinexnet`: `RetinexNet` (`forward(image) -> {"enhanced", "reflectance", "illumination",
  "illumination_adjusted"}`; any size), following the PyTorch reimplementation's network (the official TensorFlow
  weights don't fit it). `RetinexNet_Weights`: the reimplementation's release merged into one file (`REIMPL_9200`,
  default; LOL-v1 17.61 dB) and the owner's `mon_old` LOL-v1 checkpoints (`LOL_V1_PSNR`, `LOL_V1_SSIM`). Matches the
  reimplementation within 1e-6. Training losses not ported yet.
- `mon.models.lle.hvi_cidnet`: `HVICIDNet` (`forward(image) -> {"enhanced", "hvi"}`; `H, W` divisible by 8;
  `saturation_gain` / `output_gain` = the official eval gates), `rgb_to_hvi` / `hvi_to_rgb`, and nine released
  weights `HVICIDNet_Weights` (zoo files with the same tensors as the official Hugging Face releases; default
  `LOL_V1_WO_PERC`). Matches the official network; reproduces the README's LOLv1 and LOLv2-real PSNRs (±0.003
  dB). Training loss not ported yet.
- `mon.models.lle.flol`: `FLOL` (`forward(image) -> {"enhanced", "amplitude_enhanced"}`; even `H, W`, padded to 8
  inside), released weights `FLOL_Weights.LOL_V2_REAL` (default; official file, pinned URL, SHA-256) and
  `UHD_LL` (official tensors, unwrapped). BasicSR `{"params": ...}` checkpoints are unwrapped on load. Matches the
  official network; reproduces the README's LOL-v2-real 21.75 dB with the official protocol. No `kornia`
  dependency; no training loss (none released).
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

### Changed

- `mon benchmark` on a zero-shot model (`colie`, `zero_ig`, `zs_n2n`) now times its per-image fit (the registry `fit`,
  with `fit_args`), as `mon predict` runs it: latency per fit, peak memory during the fits, MACs of the whole fit
  (forward and backward, every step). `warmup`/`runs` default to 0/1 there (5/20 for a forward pass);
  `zero_shot=false` measures the network's forward pass as before. `mon benchmark model=colie` used to crash.
- `ZERO_DCE_RECIPE` trains at 512×512, as the Zero-DCE paper states (the official script resizes to 256); on SICE-ME val
  this is closer to the released weights (PSNR 13.40 vs 13.07 at 256; released 13.59).

### Fixed

- `mon benchmark data=…` kept all images of a resolution on the device at once, so each peak memory included the other
  inputs (+1.5 GiB at 8K for 5 images); images now move to the device one at a time.
- FLOL gave platform-dependent outputs (~1e-3 apart on arm64 vs x86_64): FFT bins that are real for real input
  carried ±1e-17 or exactly 0 imaginary parts, flipping their phase between ±π before the learned phase MLP. They
  are now set to exactly 0 (equal to the official code on x86); the parity literal was regenerated there.

- `mon train resume=…` on CUDA crashed (`RNG state must be a torch.ByteTensor`): the checkpoint loads onto the GPU and
  the saved RNG states are now moved back to the CPU before restoring.
- `DataLoader` workers crashed on Python 3.14 (`forkserver` pickles datasets and samples): `FolderDataset`,
  `ImageFolder` and `SampleMeta.paths` used `types.MappingProxyType`, which doesn't pickle; now `ReadOnlyDict`.

### Removed

- Unused runtime dependencies `tensordict` (samples are plain dicts) and `tensorboard` (returns with the runners).
