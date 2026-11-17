# Changelog

## 3.0.0.dev0 — Unreleased

### Added

- Packaging with uv and the `uv_build` backend; install with `uv sync`.
- Minimal runtime dependencies: `numpy`, `torch`, `torchvision`.
- Runtime dependencies: `aenum`, `fjson`, `pyyaml`, `requests`, `rich`, `tqdm`,
  `xmltodict`, `tensordict`, `omegaconf`,
  `tensorboard`.
- `opencv-python-headless` as a runtime dependency.
- `py.typed` marker (PEP 561): type checkers use `mon`'s annotations.
- Supports Python >= 3.13 (except 3.14.1), with no upper bound.
- `gui` extra (`mon[gui]`): `matplotlib` for displaying images, `PySide6-Essentials` for GUI apps.
