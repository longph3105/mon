# Changelog

## 3.0.1.dev0 — Unreleased

### Fixed

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

### Changed

- `tools/dataset_overview.py`: a dataset split's overview goes to
  `<dataset root>/overview_<split>.jpg` instead of `<split>/overview.jpg`,
  named like a folder's. A split kept in another folder is named after that
  folder (`sice_under`: `sice/overview_test_under.jpg`, beside `sice`'s
  `overview_test.jpg`). `split_dir` is replaced by `overview_path(ds, split)`.
  Move or regenerate existing overviews.
- `tools/dataset_overview.py`: 10 items per overview by default (was 20).

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
