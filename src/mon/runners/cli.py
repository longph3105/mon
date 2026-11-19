"""Command-line entry point: ``mon train``, ``mon predict``, ``mon evaluate``, ``mon benchmark``.

Usage: ``mon <command> [config.yaml] key=value ...`` (OmegaConf dotlist overrides).
"""

__all__ = [
    "main",
]

import sys
from collections.abc import Sequence

USAGE = """\
usage: mon {train,predict,evaluate,benchmark} [CONFIG.yaml] [key=value ...]

commands:
  train     Train a model with its official recipe (keys: mon.runners.trainer.TrainConfig).
  predict   Run a model over images and save its outputs (keys: mon.runners.predictor.PredictConfig).
  evaluate  Score saved predictions with image-quality metrics (keys: mon.runners.evaluator.EvaluateConfig).
  benchmark Measure params, FLOPs/MACs, latency, peak memory (keys: mon.runners.benchmarker.BenchmarkConfig).

examples:
  mon train model=zero_dce                                  # official recipe: SICE-ME, 100 epochs
  mon train model=zero_dce epochs=10 val_data=lol_v1 name=short
  mon train resume=runs/train/zero_dce/sice_me_train/2026_11_19_101500 epochs=200
  mon predict model=zero_dce data=path/to/images
  mon predict model=zero_dce++ weights=sice model_args.scale_factor=12 data=lol_v1 split=test
  mon predict model=zero_ig data=night.png fit_args.iterations=500 device=cuda:0
  mon evaluate pred=runs/predict/dccnet/lol_v1_test data=lol_v1 "metrics=[psnr,ssim,lpips]"
  mon evaluate pred=runs/predict/zero_dce/dicm_test metrics=[niqe]
  mon benchmark model=zero_dce size=[900,1200] device=cuda:0
  mon benchmark model=zero_dce++ model_args.scale_factor=12 output=zero_dce++.yaml
  mon benchmark model=colie data=testkit_efficiency device=cuda:0  # zero-shot: per-image fit time
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the command, and return the exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(USAGE, end="")
        return 0 if args else 2
    command, rest = args[0], args[1:]
    if command not in ("train", "predict", "evaluate", "benchmark"):
        print(f"mon: unknown command {command!r}\n\n{USAGE}", end="", file=sys.stderr)
        return 2
    if any(a in ("-h", "--help") for a in rest):
        print(USAGE, end="")
        return 0

    # Heavy imports only once a command runs, so `mon --help` stays instant.
    from mon.core.console import setup_logging

    path = rest.pop(0) if rest and "=" not in rest[0] else None
    setup_logging()
    try:
        _run(command, path, rest)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as err:  # config/data errors: no traceback
        print(f"mon {command}: error: {err}", file=sys.stderr)
        return 2
    return 0


def _run(command: str, path: str | None, rest: list[str]) -> None:
    from mon.core.config import load_config

    if command == "train":
        from mon.runners.trainer import train

        print(train(load_config(path, rest)))  # no schema here: the model's recipe sits under these values
    elif command == "predict":
        from mon.runners.predictor import PredictConfig, predict

        print(predict(load_config(path, rest, schema=PredictConfig)))
    elif command == "evaluate":
        from mon.runners.evaluator import EvaluateConfig, evaluate

        evaluate(load_config(path, rest, schema=EvaluateConfig))
    else:
        from mon.runners.benchmarker import BenchmarkConfig, benchmark

        benchmark(load_config(path, rest, schema=BenchmarkConfig))


if __name__ == "__main__":
    raise SystemExit(main())
