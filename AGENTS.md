# AGENTS.md — mon

Entry point for AI agents working in this repo. It takes precedence over the workspace `AGENTS.md`
for anything inside `code/mon/`; workspace rules still apply where this file is silent.

`mon` is a Python + PyTorch library for computer-vision research (named after the owner's cat).
It is a fresh rewrite of `../mon_old`, which was a uv/Poetry monorepo (`libs/`, `projects/`, `tools/`).
The rewrite is a single package with a `src/` layout.

---

## Status

The repo is a scaffold. Packaging uses **uv** with the `uv_build` backend (`pyproject.toml`, `uv.lock`).
Tests run with pytest and types are checked with mypy (both in the `dev` group). There is no linter or formatter config yet.

- Don't invent commands (`uv run …`, `pytest …`) until the config that backs them exists.
- When adding tooling, propose the choice first (package manager, build backend, linter, test runner).
  Then update the **Commands** section below in the same change.

---

## Layout

| Path           | Purpose                                                       |
|----------------|---------------------------------------------------------------|
| `src/mon/`     | The importable package. All library code lives here.          |
| `tests/`       | Tests, mirroring the `src/mon/` module tree.                  |
| `docs/`        | Long-form documentation.                                      |
| `logs/`        | Agent work logs for this repo: `logs/log_YYYY_MM_DD.md`.      |
| `CHANGELOG.md` | User-facing changes, newest first.                            |
| `README.md`    | Install, usage, citation.                                     |

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

- **Python ≥ 3.13**, PyTorch ≥ 2.x. Type-annotate public functions and classes.
- **Google-style docstrings.** Give each module a one-line docstring, and add `References:` with links to
  the paper or source code when you port or adapt something.
- No shebang (`#!/usr/bin/env python`) or encoding line (`# -*- coding: utf-8 -*-`) in package modules.
  Python 3 source is UTF-8 by default, and CLI entry points go in `[project.scripts]`. Strip both when porting.
- Registries: one `Registry` (`mon.core.registry`) per kind, defined in the package that owns it. Register with an
  explicit name when `cls.__name__.lower()` isn't the paper's name (`@MODELS.register("zero_dce++", arch=...)`).
- Enums: subclass `AliasStrEnum` (`mon.core.enums`), not `aenum`. No `"default"` member or `None` fallback;
  put defaults in signatures (`split: Split = Split.TRAIN`).
- Declare `__all__` at the top of every public module.
- Group large modules with section comments: `# --- Section Name ---`.
- Match the surrounding style. The owner aligns `=` and `:` in related assignments and signatures;
  keep that alignment and don't reformat code you aren't changing.
- Reproducibility: seed every RNG through `mon.core.runtime.seed_everything()`; save the seed it returns with the run. Pin dependencies.
  Don't add a dependency without saying why.
- Logging: library code uses `logger = get_logger(__name__)` from `mon.core.console` and never configures
  logging or prints at import. Only entry points call `setup_logging()`. Use `%`-style args: `logger.info("x=%s", x)`.
- Devices: `select_device(spec)` from `mon.core.runtime`; never fall back to CPU silently.
- Never hardcode data, weight, or output paths. Read them from config or env vars (for example `DATA_DIR`).
  Pretrained weights: `zoo_dir() / "…"` (`mon.core.path`; `MON_ZOO_DIR`, else `$DATA_DIR/zoo`).
- Tensors are `[B, C, H, W]` unless stated otherwise; document shapes in docstrings when they're not obvious.

---

## Porting From `mon_old`

- `../mon_old` is **read-only reference**. Never edit it.
- Port module by module and on purpose: clean up, drop dead code, and add tests as you go.
  Don't bulk-copy directories, and don't port `archive/` code unless asked.
- Note what you ported and from where in the work log.

---

## Workflow

1. **Ask before** creating the git repo, adding remotes, publishing packages, or changing the public API
   of code that already has callers.
2. Add or update tests for any behavior change. Mark slow or GPU-only tests with `@pytest.mark.slow` / `@pytest.mark.gpu`.
3. Add an entry to `CHANGELOG.md` for user-visible changes.
4. After significant work, append a short summary to `logs/log_YYYY_MM_DD.md` in this repo.
5. Never commit model weights, datasets, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), or secrets.
