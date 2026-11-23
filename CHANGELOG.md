# Changelog

## 3.0.1.dev0 — Unreleased

### Fixed

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
- `tools/dataset_overview.py <folder>` writes `<name>_overview.jpg` beside the
  folder, not `overview.jpg` inside it, which `mon predict data=<folder>` then
  read as an input.
