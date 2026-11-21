# AGENTS.md — mon

Rules for AI agents in this repo. They override the workspace `AGENTS.md` inside `code/mon/`; its rules apply
where this file is silent. User docs: `README.md`, `docs/`.

`mon` is a Python + PyTorch library for computer-vision research (named after the owner's cat): a single
`src/`-layout package, a fresh rewrite of the `../mon_old` monorepo.

## Session start

First reply of every session (new, after `/clear`, or after a long idle gap), before any work:

1. **Agent:** harness and version, model name and ID.
2. **Machine:** `hostname -s` and OS.
3. **Purpose:** ask what the session is for, and wait for the answer.
4. **Fetch:** ask whether to `git fetch` first. The working tree syncs between machines through Resilio, `.git`
   doesn't: on a machine whose branch is behind, `git status` shows other machines' pushed commits as local
   changes. Compare file contents with `origin/<branch>` to find real changes; realign with
   `git reset origin/<branch>` (files untouched) after the owner's OK. Never commit from a checkout behind `origin`.
5. **Resume:** read the newest entries in `logs/<host>/` (every host; before 2026-11-19: `logs/archive/`).

## Status

Version `3.0.0.dev0`, branch `3.0.0`, pre-alpha (not usable as a library yet). Datasets and weights live in the
workspace `data/` (ids, layouts, zoo naming: `data/README.md`).

| Package       | State       | Contents                                                                                                                                                                                                                                                                                                                          |
|---------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mon.core`    | Done        | Paths, file I/O, downloads, config (OmegaConf), logging, progress, enums, registry, runtime (devices, seeding), weights, size, timing                                                                                                                                                                                             |
| `mon.data`    | Done        | `io`, `labels`, `structures`, `dataset`, `loader`, `transforms`; `zoo` (`DATASETS.names()`): the `data/README.md` datasets except `sti`, with `lsrw` as `lsrw_huawei`/`lsrw_nikon`, `unpaired_lol` as its six sets, `testkit` as `testkit_*`; plus `sice_under`. Samples are dicts of `tv_tensors`; transforms are torchvision v2 |
| `mon.nn`      | In progress | Zero-reference losses, model utils, tiled inference, complexity (params, FLOPs/MACs, latency, memory); layers added per model                                                                                                                                                                                                     |
| `mon.models`  | In progress | Lazy `MODELS` index and `load`; 17 models (`MODELS.names()`): LLE with released weights and parity tests, zero-shot (`colie`, `zero_ig`, `zs_n2n`), classical (`mertens`, `tv_denoise`)                                                                                                                                           |
| `mon.metrics` | In progress | `psnr`, `ssim`, `lpips`, `niqe`, `brisque`, `nrqm`, `pi`, lazy `METRICS`; `ranking` (NTIRE 2025 LLIE composite)                                                                                                                                                                                                                   |
| `mon.runners` | In progress | `mon train` (recipes: `zero_dce`, `zero_dce++`, `sci`, `sci++`, `zero_ig`), `predict`, `evaluate`, `benchmark`                                                                                                                                                                                                                    |

**Next:** recipes for the other models (PairLIE: exposure-pair sampling; HVI-CIDNet, RetinexNet: losses ported;
FLOL, DCC-Net: wait on an LPIPS-loss home and the DCC-Net paper); more metrics; plotting. Decisions deferred to
these ports: `logs/archive/log_2026_11_17.md`, under each `core` module.

## Layout

| Path               | Purpose                                                                           |
|--------------------|-----------------------------------------------------------------------------------|
| `src/mon/core/`    | Shared infrastructure; imports nothing else from `mon`                            |
| `src/mon/data/`    | Files → samples → batches; imports only `mon.core`                                |
| `src/mon/metrics/` | Image-quality metrics, lazy `METRICS`; imports only `mon.core`                    |
| `src/mon/nn/`      | Losses, model helpers, tiling, complexity; imports only `mon.core`                |
| `src/mon/models/`  | Models by task (`lle/`, …), lazy `MODELS`, `load`; imports `mon.core`, `mon.nn`   |
| `src/mon/runners/` | Runners and the `mon` CLI; imports any `mon` package                              |
| `tools/`           | Scripts on top of `mon`, run from a checkout, not packaged: `dataset_overview.py` |
| `tests/`           | Tests, mirroring `src/mon/`                                                       |
| `docs/`            | Long-form docs (`data.md`, `efficiency.md`)                                       |
| `logs/`            | Agent work logs, `logs/<host>/log_YYYY_MM_DD.md`                                  |
| `CHANGELOG.md`     | User-facing changes, newest first                                                 |
| `README.md`        | Install, usage, citation                                                          |

Each package's overview is its docstring (`help(mon.data)`). Keep each fact in one place: link to `docs/`
instead of restating.

## Commands

| Task             | Command                                                                     |
|------------------|-----------------------------------------------------------------------------|
| Install          | `uv sync` (`--extra gui` for display)                                       |
| Add a dependency | `uv add <pkg>` (dev-only: `uv add --dev <pkg>`)                             |
| Update the lock  | `uv lock`                                                                   |
| Run in the env   | `uv run python …`                                                           |
| Build            | `uv build`                                                                  |
| Bump version     | `uv version --bump major\|minor\|patch\|alpha\|beta\|rc\|stable\|post\|dev` |
| Test             | `uv run pytest`                                                             |
| Test, fast/CPU   | `uv run pytest -m "not slow and not gpu"`                                   |
| Test, full data  | `uv run pytest -m full`                                                     |
| Type check       | `uv run mypy`                                                               |
| Lint Markdown    | `npx markdownlint-cli2 "*.md" "docs/**/*.md" "logs/**/*.md"`                |
| Check data paths | `uv run python -m mon.core.path`                                            |
| Dataset overview | `uv run python tools/dataset_overview.py <id or folder> …`                  |
| Train            | `uv run mon train model=<name> [data=<id or folder>] …`                     |
| Predict          | `uv run mon predict model=<name> data=<path or id> …`                       |
| Evaluate         | `uv run mon evaluate pred=<run dir> data=<id or folder> …`                  |
| Benchmark        | `uv run mon benchmark model=<name> [size=[H,W]] …`                          |

- Tooling: uv with `uv_build` (`pyproject.toml`, `uv.lock`); pytest and mypy (strict, `src/mon` and `tools`) in
  the `dev` group; markdownlint (`.markdownlint.yaml`, 120 columns). No Python linter or formatter yet.
- `uv run pytest` runs `tests/` plus doctests in `src/mon/`. Markers: `slow`, `gpu`, `full` (`--strict-markers`);
  `full` (whole datasets vs published numbers) is off by default: run it after changing a model or the data
  pipeline, and before releases.
- The lock must resolve for macOS arm64 and Linux x86_64 (`tool.uv.required-environments`).
- The version lives in `pyproject.toml`. From `3.0.0.dev0`, `--bump stable` releases `3.0.0`; `--bump minor`
  jumps to `3.1.0`.

## Never

- Edit `../mon_old` (read-only reference, no git; needed for parity tests and its `mon` conda env until the owner
  deletes it).
- Commit weights, datasets, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), tensor files, image or video assets, or
  secrets.
- Copy unlicensed official code into the repo, or copy pyiqa (non-commercial: a reference only, never a
  dependency).
- Run downloaded third-party code (official repos) without the owner's go-ahead, each time.
- Add `opencv-python` (only `opencv-python-headless`: both install `cv2/`), or a runtime dependency before the
  module that needs it is ported.
- Change a paper recipe when retraining from the paper (no `batch_size=auto` or other optimization changes);
  speed-ups with identical math (`cache=true`, `cache_device=device`) are fine.
- Invent commands for tooling that doesn't exist yet. To add tooling, propose the choice first, then update
  Commands in the same change.

## Ask first

Rewriting git history (amend or force-push of pushed commits), adding remotes, publishing packages, changing
the public API of code that has callers.

## Before every commit

1. Tests for any behavior change (`@pytest.mark.slow` / `gpu` where needed); `uv run pytest` and `uv run mypy`
   pass.
2. `CHANGELOG.md` entry for user-visible changes.
3. Markdown passes markdownlint.
4. A log entry after significant work (Logs).

## Code conventions

### General

- Google-style docstrings; every module has a one-line docstring, plus `References:` (paper or source links)
  when ported or adapted. Strip `mon_old`'s shebang and `# -*- coding: utf-8 -*-` lines.
- `__all__` at the top of every public module. Package namespaces re-export submodules lazily: a new public
  name goes in the package's `_EXPORTS`, `__all__` and docstring table (`tests/test_namespaces.py` checks these
  and the layering in `ALLOWED_IMPORTS`). Inside `mon`, import from submodules
  (`from mon.core.path import Path`), never from a namespace. Top-level `mon` exposes only subpackages.
- Section comments in large modules: `# --- Section Name ---`. Keep the owner's alignment of `=` and `:`.
- Configs: OmegaConf via `mon.core.config.load_config`/`save_config`; schemas are dataclasses next to their
  code; resolve devices, weights and run dirs explicitly after loading (no property setters).
- Registries: one `Registry` (`mon.core.registry`) per kind, in the package that owns it; give an explicit name
  when `cls.__name__.lower()` isn't the paper's (`register("zero_dce++", ...)`).
- Enums subclass `AliasStrEnum` (`mon.core.enums`); no `"default"` member or `None` fallback: defaults go in
  signatures (`split: Split = Split.TRAIN`).
- Seeds: `mon.core.runtime.seed_everything()`; save the returned seed with the run.
- Logging: `logger = get_logger(__name__)` (`mon.core.console`); never configure logging or print at import.
  Only entry points call `setup_logging()`.
- Devices: `select_device(spec)` (`mon.core.runtime`); never fall back to CPU silently.
- Samples and datasets must pickle (DataLoader workers; Python 3.14 `forkserver`): `mon.core.utils.ReadOnlyDict`,
  never `types.MappingProxyType`.

### Data

- A dataset subclasses `FolderDataset` (`ClassFolderDataset` for `<split>/<class>/`), sets `folder` (the
  dataset id under `datasets_dir()`) and `layout` (its family's normalized layout: `paired_layout(...)`,
  `detection_layout(...)`, `classification_layout(...)`), and registers with `register_dataset` under the same
  id, in the `mon.data.zoo` module of its family (named after an id, e.g. `lol` for `lol_v1`/`lol_v2_real`; never
  per task: tasks are tags), imported in `zoo/__init__.py`. Record known train/test overlaps in its docstring.
- Loaders: `create_dataloader` (seeded). Image/video I/O: `mon.data.io` (OpenCV), never `torchvision.io`
  codecs. `uint16` tensors: `.float()`/`v2.ToDtype` before arithmetic.
- Samples are dicts of `tv_tensors`; wrap every image-like value (`Image`/`Mask`/`Video`): v2 treats only the
  first plain tensor as an image. Class-id maps are `Mask`; continuous maps (depth, targets) are `Image`; restore
  types after arithmetic with `tv_tensors.wrap`.
- Transforms (torchvision v2) run in two stages: geometric on the whole sample, photometric only on inputs
  (`ApplyTo`), since v2 photometric ops change every `Image`. After crops, `SanitizeBoundingBoxes`.
- Stride multiples: training resizes (`ResizeToMultiple`); inference pads (`PadToMultiple`, `pad_to_multiple`)
  and crops back with `crop_to(pred, meta.orig_size)`; never resize an output back for evaluation.

### Models, losses, weights

- Models are plain `nn.Module`s, tensors in and out (`forward(image) -> Tensor` or a small dict); no TensorDict,
  `in_keys`/`out_keys` or `__call__` override; constructors take architecture arguments only. One family per
  `mon/models/<task>/<family>.py` (weights enums, models, paper loss).
- Register a model with one `MODELS.add_lazy(name, "module:Class", tasks=..., arch=...)` line in
  `mon/models/registry.py`, not a decorator (`tests/models/test_registry.py` checks the index).
- Weights: a `WeightsEnum` of `Weights("<task>/<model>/<file>.pt")` (under `zoo_dir()`; naming: `data/README.md`,
  Zoo) with `url` and `sha256`; the class sets `weights_enum: ClassVar[type[WeightsEnum]]`; load through
  `mon.models.load(name, weights=...)` and `resolve_weights(spec, Enum)`. A missing file raises, never skips.
- Large images: `mon.nn.tiling.tiled_forward`, never a per-model `forward_patch`. Stride requirement:
  `input_multiple: ClassVar[int]`.
- Classical methods are parameter-free modules. Multi-image inputs take an explicit axis (`[B, N, C, H, W]`).
  Zero-shot methods register with `zero_shot=True` and a `fit` function that fits a fresh network per image
  (`zs_n2n_denoise`); `forward` is the network only; never resize their inputs.
- Losses are separate modules named `…Loss`, returning per-sample values reduced by
  `reduction="mean"|"sum"|"none"`. Shared ones live in `mon.nn.losses.<family>`; one model's loss stays in its
  module until a second model needs it.
- Ported code follows the official code. Parity: losses against a reference copy of that code; models against
  reference values with the released weights, from a script next to the test
  (`tests/models/lle/zero_dce_reference.py`) run on the official code or `mon_old` (its `mon` conda env, when its
  code is a faithful copy), kept as literals. Record differences from the official code and `mon_old` in the module
  docstring.

### Metrics

- `Metric` modules score each image (`[B]` float64, no reduction), with `mode` (`"fr"`/`"nr"`) and
  `higher_better`, mirrored in `METRICS`. Defaults reproduce pyiqa (PSNR on RGB; SSIM and NIQE on luma rounded
  to 8-bit levels). Evaluate saved 8-bit outputs, not in-memory floats.

### Training and runners

- One loop (`mon.runners.trainer`); a model adds a `TrainRecipe` (`mon.nn.training`: step, loss, init, official
  defaults; alternating optimizers via `optimizers` + `schedule`) named by its `MODELS` entry (`train=`). Recipe
  defaults reproduce the official script (check with `tests/runners/test_trainer_parity.py`). Training resizes,
  never pads; checkpoints resume exactly.
- Runners take a dataclass config (`TrainConfig`, `PredictConfig`, `EvaluateConfig`) via `load_config` (YAML +
  dotlist), resolve it (`seed_everything`, `select_device`, run dir) and save it with the seed in the run dir.
- Run dirs: `runs_dir() / "train"` or `"predict"`, or a hand-set `"eval"`/`"benchmark"` dir (`$RUNS_DIR`, else
  `$DATA_DIR/runs`; layout: `data/README.md`, Runs). Evaluation writes next to the predictions. An existing,
  non-empty run dir raises unless `exist_ok`, and is never deleted.

### Tests

- Synthetic data only (NumPy/torch arrays, tiny files in `tmp_path`). Sample photos for manual work stay under
  `$DATA_DIR`, accessed by path.
- Regression tests on real data use the testkit (`mon.data.zoo.testkit`: 100 samples at 256×256 per dataset, one
  per task family) on the `regression_device` fixture (CUDA without TF32, else CPU); extend it when a new task
  gets data and models. Whole-dataset checks are `full`.

## Porting from `mon_old`

Port module by module and on purpose: clean up, drop dead code, add tests. Don't bulk-copy directories or port
`archive/` code unless asked. Note what you ported, and from where, in the log.

## Logs

- One file per machine: `logs/<host>/log_YYYY_MM_DD.md` (`<host>` = lowercase `hostname -s`; append). The folder
  syncs; two machines writing one file make Resilio conflict copies.
- Heading: `## HH:MM TZ · <machine> — <title>` (`date '+%H:%M %Z'`, `hostname -s`).
- After significant work, a short summary of what changed.
