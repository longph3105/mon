# Changelog

## 3.0.2.dev0 — Unreleased

### Added

- `ClampPixels` (`mon.data`): clamps float images and videos to a range
  (default `[0, 1]`); masks, boxes and integer tensors pass through.

### Changed

- Training run folders use ISO dates:
  `runs/train/<model>/<data>/<YYYY-MM-DD_HHMMSS>[_<name>]` (was
  `YYYY_MM_DD_HHMMSS`).

### Fixed

- Training transform: bicubic resizing (`resize`) left pixels outside
  `[0, 1]` (down to about −0.06 on SICE-ME at 128×128). They are now
  clamped. Models whose dynamics diverge for negative inputs (CLODE's
  curve ODE) could crash training with `underflow in dt`.
- `ColorConstancyLoss`: NaN gradients when an image's channel means are
  exactly equal (e.g. a fully saturated crop). The sum under `sqrt` is now
  clamped to `1e-12` (`SQRT_MIN`); values and gradients elsewhere are
  unchanged.
