# Changelog

## 3.0.2.dev0 — Unreleased

### Fixed

- `ColorConstancyLoss`: NaN gradients when an image's channel means are
  exactly equal (e.g. a fully saturated crop). The sum under `sqrt` is now
  clamped to `1e-12` (`SQRT_MIN`); values and gradients elsewhere are
  unchanged.
