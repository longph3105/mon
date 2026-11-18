"""Command-line entry point: ``mon predict``.

Usage: ``mon predict [config.yaml] key=value ...`` (OmegaConf dotlist overrides).
"""

__all__ = [
    "main",
]

import sys
from collections.abc import Sequence

USAGE = """\
usage: mon predict [CONFIG.yaml] [key=value ...]

Run a model over images and save its outputs (mon.runners.predictor.PredictConfig lists the keys).

examples:
  mon predict model=zero_dce data=path/to/images
  mon predict model=zero_dce++ weights=sice model_args.scale_factor=12 data=lol_v1 split=test
  mon predict model=zero_ig data=night.png fit_args.iterations=500 device=cuda:0
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the command, and return the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command != "predict":
        print(f"mon: unknown command {command!r}\n\n{USAGE}", end="", file=sys.stderr)
        return 2
    if any(a in ("-h", "--help") for a in rest):
        print(USAGE, end="")
        return 0

    # Heavy imports only once a command runs, so `mon --help` stays instant.
    from mon.core.config import load_config
    from mon.core.console import setup_logging
    from mon.runners.predictor import PredictConfig, predict

    path = rest.pop(0) if rest and "=" not in rest[0] else None
    setup_logging()
    cfg = load_config(path, rest, schema=PredictConfig)
    print(predict(cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
