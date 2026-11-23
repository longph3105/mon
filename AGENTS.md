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
| `CONVENTIONS.md` | Code conventions for `src/` and `tests/` |
| `README.md`      | Install and usage                        |

- Each package's overview is its docstring (`help(mon.data)`). Keep each fact in
  one place: link to `docs/` instead of restating.
- When a change adds a package, folder or top-level file, or new functionality
  that a row's Purpose doesn't cover (e.g. a new kind of model or metric
  family), update this table in the same commit. Keep each Purpose short: what
  the folder is for, not a list of its contents.

## Code Conventions

Before writing or changing code in `src/` or `tests/`, read and follow
[CONVENTIONS.md](CONVENTIONS.md).

## Weights

- Never commit model weights, checkpoints or run output (`*.pt`, `*.pth`,
  `*.ckpt`, `*.safetensors`, ...). Weights go in the workspace's `data/zoo/`,
  run output in `data/runs/` (see `../../data/README.md`). Check
  `git status` before committing: `.gitignore` doesn't cover every format.

## Logs

- Log work on this repo in `logs/<machine>/YYYY-MM-DD.md`, not in the
  workspace's `coord/logs/`. Machine name, times and entry format follow
  `../../coord/AGENTS.md` ("Machine name", "Times", "Logs").
- Write an entry after significant work: what changed and why, results
  (metrics, parity numbers), and what is left open.
- Each machine writes only to its own folder. Never rewrite or delete earlier
  entries; add a correction as a new entry.
- Logs up to 3.0.0 were cleared on the `3.0.1` branch; read them at the
  `v3.0.0` tag (`git show v3.0.0:logs/<machine>/YYYY-MM-DD.md`).
- Logs are committed like any other change in this repo (only when the owner
  asks), with the message `logs: <machine> <short title>`.
