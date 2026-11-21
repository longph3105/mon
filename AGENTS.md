# AGENTS.md — mon

Entry point for AI agents working in this repo. It takes precedence over the workspace `AGENTS.md`
for anything inside `code/mon/`; workspace rules still apply where this file is silent.

`mon` is a Python + PyTorch library for computer-vision research (named after the owner's cat).
It is a fresh rewrite of `../mon_old`, which was a uv/Poetry monorepo (`libs/`, `projects/`, `tools/`).
The rewrite is a single package with a `src/` layout.

---

## Session Start

At the start of every session (new, after a manual `/clear` or reset, or resumed after a long idle gap),
make your first reply, before any other work:

1. **Agent:** tool/harness and version if known, model name and ID.
2. **Machine:** `hostname -s` and OS.
3. **Purpose:** ask what this session is for; wait for the answer before starting work.
4. **Fetch:** ask the owner whether to fetch first (`git fetch`, then compare with `origin`) before reading or
   editing anything in this repo. Work is often pushed from other machines, so the local branch may be stale.
   The working tree syncs between machines through Resilio but `.git` does not: on a machine whose branch is
   behind, `git status` shows other machines' pushed commits as uncommitted changes. Compare file contents with
   `origin/<branch>` to find the real changes; `git reset origin/<branch>` (files untouched) realigns the branch
   with the owner's OK. Never commit from a checkout that is behind `origin`.

## Status

Version `3.0.0.dev0` on branch `3.0.0` (pre-alpha; not usable as a library yet).

- **Done:** `mon.core` (ported from `mon_old/libs/mon/src/mon/core`): paths, file I/O and downloads, config
  (OmegaConf), logging and progress bars, enums, registry, runtime (devices, seeding), weights, size, timing.
- **Done:** `mon.data`: `io`, `labels`, `structures`, `dataset` (`FolderDataset`, `ClassFolderDataset`, `ImageFolder`),
  `loader`, `transforms`; `zoo` has `lol_v1`, `lol_v2_real`, `loli_street`, `ntire_2025_llie`, `sice`, `sice_me`,
  `uhd_ll`, `mit_5k`, `lsrw_huawei`, `lsrw_nikon`, the six `unpaired_lol` sets, `cifar10`, `cifar100`, `mnist`,
  `fashion_mnist`, `exdark`, `darkface` and `ydld` (detection + LLE, YOLO boxes), `nightcity` (segmentation + LLE,
  Cityscapes train ids), `mot16` and `mot20` (tracking + detection, per-frame person boxes), `widerface` (face
  detection), `fisheye8k` (fisheye road objects), and the `testkit_*` test fixtures (`testkit_efficiency`: a 256²–8K
  resolution ladder for `mon benchmark`).
  Samples are dicts of `torchvision.tv_tensors`; transforms are torchvision v2.
- **In progress:** `mon.nn`: zero-reference losses, model utils, tiled inference, complexity (params, FLOPs/MACs,
  latency, peak memory) done; layers added per model.
- **In progress:** `mon.models`: lazy `MODELS` index and `load` done; `lle.zero_dce` (Zero-DCE, Zero-DCE++),
  `lle.sci` (SCI, SCI++), `lle.sgz` (SGZ), `lle.pairlie`, `lle.clode`, `lle.dccnet`, `lle.flol`, `lle.hvi_cidnet`,
  `lle.retinexnet`, `lle.ruas` ported with released weights and parity tests; `lle.colie` (zero-shot, `mon_old` parity);
  `lle.zero_ig` (zero-shot, official dataset-trained weights; official inference, loss and gradient parity);
  `mef.mertens` (exposure fusion, classical) with OpenCV parity tests; `denoise.tv_denoise` (scikit-image
  parity) and `denoise.zs_n2n` (zero-shot, `mon_old` parity).
- **In progress:** `mon.metrics`: `psnr`, `ssim`, `lpips` (full-reference), `niqe`, `brisque`, `nrqm`, `pi`
  (no-reference), lazy `METRICS` index; defaults follow pyiqa (parity tests), which is not a dependency;
  `ranking`: composite leaderboard ranks across methods (NTIRE 2025 LLIE weights).
- **In progress:** `mon.runners`: `trainer` (`mon train`; recipes for `zero_dce`, `zero_dce++`, `sci`, `sci++`,
  `zero_ig`), `predictor` (`mon predict`), `evaluator` (`mon evaluate`), `benchmarker` (`mon benchmark`).
- **Data:** datasets and weights live in the workspace `data/` (ids, layouts, zoo naming: `data/README.md`).
- **Next:** recipes for the other models (PairLIE: exposure-pair sampling; HVI-CIDNet and RetinexNet, whose losses
  are ported; FLOL and DCC-Net losses wait on an LPIPS-loss home and the DCC-Net paper); more metrics; plotting.
  Decisions deferred to these ports are recorded in `logs/archive/log_2026_11_17.md` under each
  `core` module's entry.
- **Tooling:** uv with the `uv_build` backend (`pyproject.toml`, `uv.lock`); pytest and mypy (strict) in the
  `dev` group. Markdown: markdownlint (`.markdownlint.yaml`, 120 columns). No Python linter or formatter yet.
- **Resuming work:** read the latest "Session handoff" section at the end of the newest log (`logs/*/log_*.md`;
  up to 2026-11-19, `logs/archive/log_*.md`).
- **Work log entries:** one file per machine, `logs/<host>/log_YYYY_MM_DD.md` (`<host>` = lowercase
  `hostname -s`; the folder syncs, and two machines writing one file make Resilio conflict copies). Head each
  with the local time and the machine the agent runs on: `## HH:MM TZ · <machine> — <title>`
  (`date '+%H:%M %Z'`, `hostname -s`).
- Don't invent commands (`uv run …`, `pytest …`) until the config that backs them exists.
- When adding tooling, propose the choice first (package manager, build backend, linter, test runner).
  Then update the **Commands** section below in the same change.

---

## Layout

| Path               | Purpose                                                                                                               |
|--------------------|-----------------------------------------------------------------------------------------------------------------------|
| `docs/`            | Long-form documentation.                                                                                              |
| `logs/`            | Agent work logs, one file per machine: `logs/<host>/log_YYYY_MM_DD.md` (to 2026-11-19: `logs/archive/`).              |
| `src/mon/`         | The importable package. All library code lives here.                                                                  |
| `src/mon/core/`    | Shared infrastructure; no deps on the rest of `mon`. Overview: `help(mon.core)`.                                      |
| `src/mon/data/`    | Files → samples → batches; may import only `mon.core`. Overview: `help(mon.data)`.                                    |
| `src/mon/metrics/` | Image-quality metrics and the lazy `METRICS` index; may import only `mon.core`. Overview: `help(mon.metrics)`.        |
| `src/mon/models/`  | Models by task (`lle/`, …), lazy `MODELS` index, `load`; imports `mon.core`, `mon.nn`. Overview: `help(mon.models)`.  |
| `src/mon/nn/`      | Losses, model helpers, tiled inference, complexity; imports only `mon.core`. Overview: `help(mon.nn)`.                |
| `src/mon/runners/` | Runners (train, predict, evaluate, benchmark), `mon` CLI; imports any `mon` package. Overview: `help(mon.runners)`.   |
| `tools/`           | Scripts on top of `mon`, not part of the package (run from a checkout): `dataset_overview.py`.                        |
| `tests/`           | Tests, mirroring the `src/mon/` module tree.                                                                          |
| `CHANGELOG.md`     | User-facing changes, newest first.                                                                                    |
| `README.md`        | Install, usage, citation.                                                                                             |

Keep each fact in one place: link to `docs/` instead of restating it here or in `README.md`.

---

## Commands

| Task              | Command                                                      |
|-------------------|--------------------------------------------------------------|
| Install           | `uv sync`                                                    |
| Install + display | `uv sync --extra gui`                                        |
| Add a dependency  | `uv add <pkg>`                                               |
| Update the lock   | `uv lock`                                                    |
| Run in the env    | `uv run python …`                                            |
| Build sdist/wheel | `uv build`                                                   |
| Bump version      | `uv version --bump <part>`                                   |
| Test              | `uv run pytest`                                              |
| Test, fast/CPU    | `uv run pytest -m "not slow and not gpu"`                    |
| Test, full data   | `uv run pytest -m full`                                      |
| Type check        | `uv run mypy`                                                |
| Dataset overview  | `uv run python tools/dataset_overview.py <id or folder> …`   |
| Check data paths  | `uv run python -m mon.core.path`                             |
| Train             | `uv run mon train model=<name> [data=<id or folder>] …`      |
| Predict           | `uv run mon predict model=<name> data=<path or id> …`        |
| Evaluate          | `uv run mon evaluate pred=<run dir> data=<id or folder> …`   |
| Benchmark         | `uv run mon benchmark model=<name> [size=[H,W]] …`           |
| Lint Markdown     | `npx markdownlint-cli2 "*.md" "docs/**/*.md" "logs/**/*.md"` |

- The lock must resolve for macOS arm64 and Linux x86_64 (`tool.uv.required-environments`).
- OpenCV is `opencv-python-headless` only; never add `opencv-python` (both install `cv2/` and clobber
  each other). `cv2.imshow` doesn't work on Linux, so show images with matplotlib (`gui` extra).
- The version lives in `pyproject.toml` (`uv_build` has no VCS versioning). Bump it with
  `uv version --bump major|minor|patch|alpha|beta|rc|stable|post|dev`; `uv version` prints it.
  From a pre-release (`3.0.0.dev0`), use `--bump stable` to release `3.0.0`; `--bump minor` jumps to `3.1.0`.
- Keep runtime deps minimal: add one only when the module that needs it is ported.
- `uv run pytest` runs `tests/` plus doctests in `src/mon/`. Markers: `slow`, `gpu`, `full` (`--strict-markers`).
  `full` tests (whole datasets vs published numbers) are off by default (`addopts`); run them with `-m full` after
  changing a model or the data pipeline, and before releases.
- Dev-only tools go in `[dependency-groups] dev` (`uv add --dev <pkg>`), never in runtime deps.
- `uv run mypy` checks `src/mon` and `tools` in strict mode; keep it clean. Add `types-*` stub packages to `dev` when needed.
- Markdown must pass markdownlint (`.markdownlint.yaml`). No Python lint or format commands yet (see **Status**).

---

## Code Conventions

- **Google-style docstrings.** Every module gets a one-line docstring; add `References:` with links to the paper
  or source code when you port or adapt something.
- Configs: OmegaConf via `mon.core.config.load_config`/`save_config`. Schemas are dataclasses next to the code that
  consumes them; resolve devices, weights, and run dirs explicitly after loading (no property setters). Save the
  resolved config into each run directory.
- Registries: one `Registry` (`mon.core.registry`) per kind, defined in the package that owns it. Register with an
  explicit name when `cls.__name__.lower()` isn't the paper's name (`register("zero_dce++", ...)`). Models are not
  decorated: each gets one `MODELS.add_lazy(name, "module:Class", tasks=..., arch=...)` line in
  `mon/models/registry.py`, so listing models imports no model code (`tests/models/test_registry.py` checks the index).
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
- Pretrained weights: declare a `WeightsEnum` of `Weights("<task>/<model>/<file>.pt")` (path under `zoo_dir()`; layout
  and file naming: workspace `data/README.md`, Zoo) per model with `url` and `sha256`, and accept user input
  through `resolve_weights(spec, Enum)`; a missing file must raise, never skip loading.
- Datasets: subclass `FolderDataset` (or `ClassFolderDataset` for `<split>/<class>/` folders), set `folder` (the dataset
  id, under `datasets_dir()`) and `layout` (the normalized layout of its task family, e.g. `paired_layout(...)`,
  `detection_layout(...)`, `classification_layout(...)`; ids and layouts: workspace `data/README.md`), register with
  `register_dataset` under the same id in the `mon.data.zoo` module of its dataset family (one module per family, named
  after its id, e.g. `lol` for `lol_v1`/`lol_v2_real`; never one per task: tasks are the `register_dataset` tags), and
  import that module in `zoo/__init__.py`. Record known train/test overlaps with other datasets in the docstring. Build
  loaders with `create_dataloader` (seeded). `uint16` tensors (16-bit images) support only some torch ops: convert with
  `.float()`/`v2.ToDtype` before arithmetic.
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
- Metrics are `Metric` modules scoring each image (`[B]` float64, no reduction) with class attributes `mode`
  (`"fr"`/`"nr"`) and `higher_better`, mirrored in their `METRICS` entry. Defaults reproduce pyiqa, which
  `mon_old` used (PSNR on RGB; SSIM and NIQE on luma rounded to 8-bit levels); pyiqa is non-commercial, so it
  is only a reference: never a dependency, never copied. Evaluate saved 8-bit outputs, not in-memory floats.
- Training: one generic loop (`mon.runners.trainer`); a model contributes a `TrainRecipe` (`mon.nn.training`: step,
  loss, init, official defaults; alternating optimizers via `optimizers` + `schedule`) named by its `MODELS` entry
  (`train=`). Recipe defaults reproduce the official script and sit under the user's config; check a new recipe's
  loop against the official script (see `tests/runners/test_trainer_parity.py`). Training resizes, never pads;
  checkpoints resume exactly.
- Retraining a model from its paper always uses the recipe as is (owner's rule): no `batch_size=auto` or other
  changes to the optimization; speed-ups that keep the math identical (`cache=true`, `cache_device=device`) are fine.
- Samples and datasets cross process boundaries (DataLoader workers; Python 3.14 uses `forkserver`, which pickles):
  keep them picklable, e.g. `mon.core.utils.ReadOnlyDict`, never `types.MappingProxyType`.
- Runners take a dataclass config (`PredictConfig`, `EvaluateConfig`, `TrainConfig`) loaded with `load_config`
  (YAML + dotlist), resolve it explicitly (`seed_everything`, `select_device`, run directory), and save the resolved
  config with the seed in the run directory (evaluation writes its files next to the predictions). Outputs go under
  `runs_dir() / "train"` or `runs_dir() / "predict"` (`$RUNS_DIR`, else `$DATA_DIR/runs`; layout: workspace
  `data/README.md`, Rules); an existing, non-empty run directory raises unless `exist_ok`, and is never deleted. A
  model's stride requirement is `input_multiple: ClassVar[int]` (runners pad to it); a zero-shot model's fit
  function is the registry's `fit`.
- Models: plain `nn.Module`, tensors in and out (`forward(image) -> Tensor`, or a small dict of named outputs);
  no TensorDict, `in_keys`/`out_keys`, or `__call__` override. The runner maps sample dicts to model inputs; a
  model's paper loss is a separate module, not a method. Large-image inference uses
  `mon.nn.tiling.tiled_forward`, never a per-model `forward_patch`. Constructors take architecture arguments only;
  pretrained weights go through `mon.models.load(name, weights=...)`, so a model class sets
  `weights_enum: ClassVar[type[WeightsEnum]]`. Put a model family in `mon/models/<task>/<family>.py` (weights enums,
  models, paper loss); loss class names end in `Loss`. Classical methods are parameter-free modules (`load(name)`
  loads nothing); multi-image inputs take an explicit axis (`[B, N, C, H, W]` for an exposure stack). Zero-shot
  methods (fitted per input at test time) register with `zero_shot=True` and provide a function that fits a fresh
  network per image (e.g. `zs_n2n_denoise`); their `forward` is the network only. Never resize their inputs.
- Losses: shared ones live in `mon.nn.losses.<family>` (e.g. `zero_reference`); a loss used by one model stays in
  that model's module until a second model needs it. Every loss returns per-sample values reduced by
  `reduction="mean"|"sum"|"none"`.
- Losses and models ported from a paper follow its official code; test parity against a reference copy of
  that code (losses) or, for models, against reference values with the released weights: a script next to the
  test (e.g. `tests/models/lle/zero_dce_reference.py`) computes output statistics on a deterministic input from
  the official code or `mon_old` (in its `mon` conda env), and the test keeps them as a literal. Don't copy
  unlicensed official code into the repo and don't commit tensor files. Record known differences from the
  official code and from `mon_old` in the module docstring. Running downloaded third-party code (official repos)
  needs the owner's explicit go-ahead each time; generate reference values with `mon_old` when its code is a
  faithful copy (check by reading), and compare by reading otherwise.

---

## Porting From `mon_old`

- `../mon_old` is **read-only reference**. Never edit it. It has no git any more (removed 2026-11-18) and the owner
  will delete it; until then it is needed for parity tests (its `mon` conda env too).
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
   Regression tests on real data use the testkit (`mon.data.zoo.testkit`: 100 samples at 256×256 per dataset, one per
   task family) on the `regression_device` fixture (CUDA without TF32, else CPU); whole-dataset checks are `full`.
   Extend the testkit when a new task gets data and models.
3. Add an entry to `CHANGELOG.md` for user-visible changes.
4. After significant work, append a short summary to `logs/<host>/log_YYYY_MM_DD.md` in this repo.
5. Never commit model weights, datasets, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), or secrets.
6. Tests use synthetic data (NumPy/torch arrays, tiny files in `tmp_path`); never commit image or video
   assets. Sample photos for manual work belong under `$DATA_DIR`, accessed by path, not constants.
