# AGENTS.md - mon

Rules for AI agents in this repo. These add to the workspace `AGENTS.md`
(`../../AGENTS.md`, from the
[longph3105/workspace](https://github.com/longph3105/workspace) repo on GitHub);
follow both.

## Layout

| Path             | Purpose                                  |
|------------------|------------------------------------------|
| `src/mon/`       | The `mon` package                        |
| `├ core/`        | Shared infrastructure                    |
| `├ data/`        | Datasets, loaders, transforms            |
| `├ metrics/`     | Image-quality metrics                    |
| `├ nn/`          | Losses and model helpers                 |
| `├ models/`      | Models by task                           |
| `└ runners/`     | Train, predict, evaluate, benchmark; CLI |
| `tools/`         | Scripts on top of `mon`                  |
| `tests/`         | Tests, mirroring `src/mon/`              |
| `docs/`          | Long-form docs                           |
| `logs/`          | Agent work logs                          |
| `pyproject.toml` | Package metadata and tool settings       |
| `uv.lock`        | Locked dependencies                      |
| `CHANGELOG.md`   | User-facing changes                      |
| `README.md`      | Install and usage                        |

- Each package's overview is its docstring (`help(mon.data)`). Keep each fact in
  one place: link to `docs/` instead of restating.
- When a change adds a package, folder or top-level file, or new functionality
  that a row's Purpose doesn't cover (e.g. a new kind of model or metric
  family), update this table in the same commit. Keep each Purpose short: what
  the folder is for, not a list of its contents.

## General Code Conventions

- Google-style docstrings; every module has a one-line docstring, plus
  `References:` (paper or source links) when ported or adapted. Strip
  `mon_old`'s shebang and `# -*- coding: utf-8 -*-` lines.
- `__all__` at the top of every public module. Package namespaces re-export
  submodules lazily: a new public name goes in the package's `_EXPORTS`,
  `__all__` and docstring table (`tests/test_namespaces.py` checks these and the
  layering in `ALLOWED_IMPORTS`). Inside `mon`, import from submodules
  (`from mon.core.path import Path`), never from a namespace. Top-level `mon`
  exposes only subpackages.
- Section comments in large modules: `# --- Section Name ---`. Keep the owner's
  alignment of `=` and `:`.
- Configs: OmegaConf via `mon.core.config.load_config`/`save_config`; schemas
  are dataclasses next to their code; resolve devices, weights and run dirs
  explicitly after loading (no property setters).
- Registries: one `Registry` (`mon.core.registry`) per kind, in the package that
  owns it; give an explicit name when `cls.__name__.lower()` isn't the paper's
  (`register("zero_dce++", ...)`).
- Enums subclass `AliasStrEnum` (`mon.core.enums`); no `"default"` member or
  `None` fallback: defaults go in signatures (`split: Split = Split.TRAIN`).
- Seeds: `mon.core.runtime.seed_everything()`; save the returned seed with the
  run.
- Logging: `logger = get_logger(__name__)` (`mon.core.console`); never configure
  logging or print at import. Only entry points call `setup_logging()`.
- Devices: `select_device(spec)` (`mon.core.runtime`); never fall back to CPU
  silently.
- Samples and datasets must pickle (DataLoader workers; Python 3.14
  `forkserver`): `mon.core.utils.ReadOnlyDict`, never `types.MappingProxyType`.
