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

## GPU

- One GPU job at a time to prevent OOM. Before starting training, a
  benchmark or GPU tests, check that nothing else is using the GPU
  (`nvidia-smi`), and wait or ask the owner if something is.

## Logs

- Log work on this repo in the workspace log,
  `../../coord/logs/<machine>/YYYY-MM-DD.md`, with `Scope: code/mon`.
  Format and rules: `../../coord/AGENTS.md` ("Logs").
- What changed in a version goes in `CHANGELOG.md`; results (metrics, parity
  numbers) go in the workspace wiki (`wiki/wiki/experiments/`), and the log
  entry names the page.
- The repo no longer keeps a `logs/` folder. Old logs are in git: up to
  3.0.0 at `v3.0.0`, 3.0.1 at `v3.0.1`, and 3.0.2 up to 2026-11-27 at
  `9977c1e` (e.g. `git show 9977c1e:logs/<machine>/YYYY-MM-DD.md`).
