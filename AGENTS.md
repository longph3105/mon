# AGENTS.md — mon

Entry point for AI agents working in this repo. It takes precedence over the workspace `AGENTS.md`
for anything inside `code/mon/`; workspace rules still apply where this file is silent.

`mon` is a Python + PyTorch library for computer-vision research (named after the owner's cat).
It is a fresh rewrite of `../mon_old`, which was a uv/Poetry monorepo (`libs/`, `projects/`, `tools/`).
The rewrite is a single package with a `src/` layout.

---

## Status

Version `3.0.0.dev0` on branch `3.0.0` (pre-alpha; not usable as a library yet).

- **Done:** `mon.core` (ported from `mon_old/libs/mon/src/mon/core`): paths, file I/O and downloads, config
  (OmegaConf), logging and progress bars, enums, registry, runtime (devices, seeding), weights, size, timing.
- **In progress:** `mon.data`: `io`, `labels`, `structures`, `dataset`, `loader`, `transforms` done; `zoo`
  has `lol_v1`, `lol_v2_real`. Samples are dicts of `torchvision.tv_tensors`; transforms are torchvision v2.
- **In progress:** `mon.nn`: zero-reference losses, model utils, tiled inference done; layers added per model.
- **Next:** `mon.models` (lazy registry, weights enums, Zero-DCE first), runners (train/predict configs, run
  directories), plotting. Decisions deferred to these ports are recorded in `logs/log_2026_11_17.md` under each
  `core` module's entry.
- **Tooling:** uv with the `uv_build` backend (`pyproject.toml`, `uv.lock`); pytest and mypy (strict) in the
  `dev` group. No linter or formatter config yet.
- **Resuming work:** read the latest "Session handoff" section at the end of the newest `logs/log_*.md`.
- Don't invent commands (`uv run …`, `pytest …`) until the config that backs them exists.
- When adding tooling, propose the choice first (package manager, build backend, linter, test runner).
  Then update the **Commands** section below in the same change.

---

## Layout

| Path            | Purpose                                                                                    |
|-----------------|--------------------------------------------------------------------------------------------|
| `src/mon/`      | The importable package. All library code lives here.                                       |
| `src/mon/core/` | Shared infrastructure; no deps on the rest of `mon`. Overview: `help(mon.core)`.           |
| `src/mon/data/` | Files → samples → batches; may import only `mon.core`. Overview: `help(mon.data)`.         |
| `src/mon/nn/`   | Losses, model helpers, tiled inference; imports only `mon.core`. Overview: `help(mon.nn)`. |
| `tests/`        | Tests, mirroring the `src/mon/` module tree.                                               |
| `docs/`         | Long-form documentation.                                                                   |
| `logs/`         | Agent work logs for this repo: `logs/log_YYYY_MM_DD.md`.                                   |
| `CHANGELOG.md`  | User-facing changes, newest first.                                                         |
| `README.md`     | Install, usage, citation.                                                                  |

Keep each fact in one place: link to `docs/` instead of restating it here or in `README.md`.

---

## Commands

| Task              | Command                                   |
|-------------------|-------------------------------------------|
| Install           | `uv sync`                                 |
| Install + display | `uv sync --extra gui`                     |
| Add a dependency  | `uv add <pkg>`                            |
| Update the lock   | `uv lock`                                 |
| Run in the env    | `uv run python …`                         |
| Build sdist/wheel | `uv build`                                |
| Bump version      | `uv version --bump <part>`                |
| Test              | `uv run pytest`                           |
| Test, fast/CPU    | `uv run pytest -m "not slow and not gpu"` |
| Type check        | `uv run mypy`                             |
| Check data paths  | `uv run python -m mon.core.path`          |

- The lock must resolve for macOS arm64 and Linux x86_64 (`tool.uv.required-environments`).
- OpenCV is `opencv-python-headless` only; never add `opencv-python` (both install `cv2/` and clobber
  each other). `cv2.imshow` doesn't work on Linux, so show images with matplotlib (`gui` extra).
- The version lives in `pyproject.toml` (`uv_build` has no VCS versioning). Bump it with
  `uv version --bump major|minor|patch|alpha|beta|rc|stable|post|dev`; `uv version` prints it.
  From a pre-release (`3.0.0.dev0`), use `--bump stable` to release `3.0.0`; `--bump minor` jumps to `3.1.0`.
- Keep runtime deps minimal: add one only when the module that needs it is ported.
- `uv run pytest` runs `tests/` plus doctests in `src/mon/`. Markers: `slow`, `gpu` (`--strict-markers`).
- Dev-only tools go in `[dependency-groups] dev` (`uv add --dev <pkg>`), never in runtime deps.
- `uv run mypy` checks `src/mon` in strict mode; keep it clean. Add `types-*` stub packages to `dev` when needed.
- No lint or format commands yet (see **Status**).

---

## Code Conventions

- **Google-style docstrings.** Every module gets a one-line docstring; add `References:` with links to the paper
  or source code when you port or adapt something.
- Configs: OmegaConf via `mon.core.config.load_config`/`save_config`. Schemas are dataclasses next to the code that
  consumes them; resolve devices, weights, and run dirs explicitly after loading (no property setters). Save the
  resolved config into each run directory.
- Registries: one `Registry` (`mon.core.registry`) per kind, defined in the package that owns it. Register with an
  explicit name when `cls.__name__.lower()` isn't the paper's name (`@MODELS.register("zero_dce++", arch=...)`).
- Enums: subclass `AliasStrEnum` (`mon.core.enums`), not `aenum`. No `"default"` member or `None` fallback;
  put defaults in signatures (`split: Split = Split.TRAIN`).
- Declare `__all__` at the top of every public module.
- Package namespaces re-export their submodules' `__all__` lazily. When adding a public name, add it to the
  package's `_EXPORTS` and `__all__` and to its docstring overview table (`tests/test_namespaces.py` enforces
  these and the layering in `ALLOWED_IMPORTS`). Code inside `mon` imports from submodules
  (`from mon.core.path import Path`), never from a package namespace. Top-level `mon` exposes only subpackages.
- Group large modules with section comments: `# --- Section Name ---`. The owner aligns `=` and `:` in related
  assignments and signatures; keep that alignment.
- Seed every RNG through `mon.core.runtime.seed_everything()` and save the seed it returns with the run.
- Logging: library code uses `logger = get_logger(__name__)` from `mon.core.console` and never configures
  logging or prints at import. Only entry points call `setup_logging()`.
- Devices: `select_device(spec)` from `mon.core.runtime`; never fall back to CPU silently.
- Pretrained weights: declare a `WeightsEnum` of `Weights("zoo/relative/path.pt")` per model and accept user input
  through `resolve_weights(spec, Enum)`; a missing file must raise, never skip loading.
- Datasets: subclass `FolderDataset`, set `folder` (the dataset id, under `datasets_dir()`) and `layout`
  (the normalized layout of its task family, e.g. `paired_layout(...)`; ids and layouts: workspace
  `data/README.md`), register with `register_dataset` under the same id in a `mon.data.zoo` module, and import
  that module in `zoo/__init__.py`. Record known train/test overlaps with other datasets in the docstring.
  Build loaders with `create_dataloader` (seeded). `uint16` tensors (16-bit images) support only some torch ops:
  convert with `.float()`/`v2.ToDtype` before arithmetic.
- Image/video I/O: `mon.data.io` (OpenCV). Don't use `torchvision.io` codecs: deprecated in torchvision 0.29.
- Samples are dicts. Wrap every image-like value explicitly (`tv_tensors.Image`/`Mask`/`Video`); v2 treats only
  the first *plain* tensor as an image and silently skips the rest. Class-id maps are `Mask` (nearest resize);
  continuous maps (depth, targets) are `Image`. Restore types after arithmetic with `tv_tensors.wrap`.
- Transforms run in two stages: geometric on the whole sample, photometric only on input keys (via `ApplyTo`):
  v2 photometric ops change *every* `Image`, including ground-truth targets and depth. After crops, use
  `SanitizeBoundingBoxes` so labels drop with their boxes.
- Stride-multiple sizes: training resizes (`ResizeToMultiple`); inference pads (`PadToMultiple`, or
  `pad_to_multiple` on a batch) and crops predictions back with `crop_to(pred, meta.orig_size)`. Predictors
  default to padding; never resize an output back for evaluation.
- Models: plain `nn.Module`, tensors in and out (`forward(image) -> Tensor`, or a small dict of named outputs);
  no TensorDict, `in_keys`/`out_keys`, or `__call__` override. The runner maps sample dicts to model inputs; a
  model's paper loss is a separate module, not a method. Large-image inference uses
  `mon.nn.tiling.tiled_forward`, never a per-model `forward_patch`.
- Losses: shared ones live in `mon.nn.losses.<family>` (e.g. `zero_reference`); a loss used by one model stays in
  that model's module until a second model needs it. Every loss returns per-sample values reduced by
  `reduction="mean"|"sum"|"none"`.
- Losses and models ported from a paper follow its official code; test parity against a reference copy of
  that code (losses) or against `mon_old` outputs with the same weights (models, `slow`; needs the old
  `mon` conda env). Record known differences between `mon_old` copies in the docstring.

---

## Porting From `mon_old`

- `../mon_old` is **read-only reference**. Never edit it.
- Port module by module and on purpose: clean up, drop dead code, and add tests as you go.
  Don't bulk-copy directories, and don't port `archive/` code unless asked.
- Note what you ported and from where in the work log.
- Strip the shebang and `# -*- coding: utf-8 -*-` lines that `mon_old` modules start with.

---

## Workflow

1. **Ask before** rewriting git history (amend/force-push of pushed commits), adding remotes, publishing
   packages, or changing the public API of code that already has callers.
2. Add or update tests for any behavior change. Mark slow or GPU-only tests with `@pytest.mark.slow` /
   `@pytest.mark.gpu`. Before committing, `uv run pytest` and `uv run mypy` must pass.
3. Add an entry to `CHANGELOG.md` for user-visible changes.
4. After significant work, append a short summary to `logs/log_YYYY_MM_DD.md` in this repo.
5. Never commit model weights, datasets, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), or secrets.
6. Tests use synthetic data (NumPy/torch arrays, tiny files in `tmp_path`); never commit image or video
   assets. Sample photos for manual work belong under `$DATA_DIR`, accessed by path, not constants.
