# Changelog

## 3.0.1.dev0 — Unreleased

### Fixed

- Zero-IG trains like the official `train.py`, which never switches back to
  training mode after its epoch-0 test pass: batch norm runs in eval mode
  after the first epoch (`mon train model=zero_ig`) or the first step
  (`zero_ig_enhance`, `mon predict model=zero_ig`). Opt out with
  `eval_mode_after=null` (`mon train`) or `fit_args.eval_mode_after=null`
  (`mon predict`). New `TrainConfig.eval_mode_after` for any recipe;
  validation now leaves the model in the mode it found.
- `SCI()` initializes like the official `CVPR/train.py`: each of the six
  sub-blocks once, in the script's order. `self.apply` also re-drew the shared
  `conv`/`convs` blocks through `blocks` (17 initializations of 11 layers), so
  `mon train model=sci` started from different weights than the official
  script (training losses about 20% off). Released weights are unaffected.
- `HVICIDNetLoss`: the perceptual term compares pre-ReLU VGG-19 features, as
  the official extractor (which copies each tap) does. torchvision's next
  in-place ReLU overwrote `conv1_2`, `conv2_2` and `conv3_4`, so the term used
  post-ReLU features for those three layers.
- `mon train resume=<run> epochs=<more>` with `scheduler.name=cosine` anneals
  over the new `epochs`. It used to keep the saved `T_max`, so the LR reached
  `eta_min` at the old end and then rose again.
- `mon evaluate` and `mon benchmark data=...` scale 16-bit PNG/TIFF images by
  65535. They used to divide every image by 255.
- `mon evaluate` no longer stops when a metric can't score one image (for
  example, an image smaller than SSIM's window or NIQE's blocks). The cell is
  left empty and a warning is logged. `metrics.yaml` lists the image under
  `failed` and gives each metric's `scored` count; means cover scored images
  only.
- `mon` prints an error line (exit code 2) for a `run_dir` that is a file,
  instead of a `NotADirectoryError` traceback.
- `tools/dataset_overview.py <folder>` writes `overview_<name>.jpg` beside the
  folder, not `overview.jpg` inside it, which `mon predict data=<folder>` then
  read as an input.
- `testkit_efficiency` is tagged `Task.BENCHMARK` instead of `Task.LLE`, so
  `DATASETS.names(tasks=Task.LLE)` no longer lists it.
- `mon evaluate` saves the device it used in `metrics.yaml` (it saved the
  spec, e.g. `auto`).
- `HVICIDNet_Weights`: each entry's `meta["source_sha256"]` is the full
  SHA-256 of its Hugging Face file (it was cut to 16 digits).
- `RUAS.weights_enum` is annotated `ClassVar`, like every other model's.
- `python -m mon.core.path` lists the runs directory (`$RUNS_DIR`, else
  `$DATA_DIR/runs`) next to data, datasets and zoo. An unset one fails the
  check; a missing one doesn't, since runners create it.

### Changed

- `tools/dataset_overview.py`: a dataset split's overview goes to
  `<dataset root>/overview_<split>.jpg` instead of `<split>/overview.jpg`,
  named like a folder's. A split kept in another folder is named after that
  folder (`sice_under`: `sice/overview_test_under.jpg`, beside `sice`'s
  `overview_test.jpg`). `split_dir` is replaced by `overview_path(ds, split)`.
  Move or regenerate existing overviews.
- `tools/dataset_overview.py`: 10 items per overview by default (was 20).
- New `mon.nn.losses.retinex.EdgeAwareSmoothnessLoss`: SCI's edge-aware
  smoothness, now shared by `SCILoss` and `ZeroIGLoss` (each had its own
  copy). `ZeroIGLoss`, `CoLIELoss` and CLODE use `IlluminationSmoothnessLoss`
  for Zero-DCE's TV term, and CLODE uses `mon.models.denoise.zs_n2n.pair_downsample`.
  Values are unchanged.
- `CONFIG_EXTENSIONS` no longer includes `.txt`, so `Path.is_config_file()` is
  false for `classes.txt` and YOLO label files.

### Deprecated

- Unused in `mon`, to be removed in 3.1 (access warns with
  `DeprecationWarning`): the enums `RunMode`, `Strategy` and `BBoxFormat`; the
  constants `ANNS_DIR`, `DEBUG_DIR`, `DEPTH_DIR`, `IMAGES_DIR`, `LABELS_DIR`,
  `DEFAULT_CKPT_EXT`, `DEFAULT_IMAGE_EXT`, `DEFAULT_LABEL_EXT` and
  `DEFAULT_WEIGHTS_EXT`.

### Docs

- CONVENTIONS: a weights entry has a `url` only when the zoo file is the
  released file itself (a converted file gives `source` and the full
  `source_sha256`); `evaluate` and `benchmark` save their config in their
  result file; a dataset's `folder` may be a subfolder of its archive
  (`lsrw/huawei`, `testkit/lle`) or shared with another dataset (`sice_under`).
- Docstrings: the detection datasets' class ids (the `classes` list, in the
  order of `classes.txt`, which isn't read); the `testkit_mef` and
  `testkit_efficiency` stems (`sice_<scene>/...`); three stale pointers
  (`transforms`, `metrics.registry`, `fileio`).

### Tests

- Training-loop parity with the official scripts for `zero_dce++`, `sci` and
  `zero_ig` (one epoch: the official Zero-IG loop switches to eval mode after
  it; see `mon.models.lle.zero_ig`), next to `zero_dce` and `sci++`.
- Loss parity with the official code for `HVICIDNetLoss`, `RetinexNetLoss` and
  `PairLIELoss`, gradients included.
- `zero_ig` (its three released weights) in the testkit regression.
- The testkit builders on synthetic sources (every part, `--part`
  rebuilds, the `python -m mon.data.zoo` CLI); LR schedulers; `runs_dir` and
  `datasets_dir`; `mon evaluate` with real no-reference metrics; 16-bit
  inputs in `mon benchmark`.
- Every weights entry has a full `sha256`, full `source_sha256` values, and a
  `url` or a `meta["source"]`.
