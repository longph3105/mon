"""Runners: drive models over data end to end (prediction, evaluation; training later).

Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    =========  ===================================================================  ========================
    Module     Purpose                                                              Key names
    =========  ===================================================================  ========================
    cli        Command-line entry point: ``mon predict`` and ``mon evaluate``.      main
    evaluator  Score saved predictions against targets with image-quality metrics.  EvaluateConfig, evaluate
    predictor  Run a model over images and save its outputs.                        PredictConfig, predict
    =========  ===================================================================  ========================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("cli", "evaluator", "predictor")

# Public name -> submodule. Keep in sync with each submodule's ``__all__``.
_EXPORTS: dict[str, str] = {
    # cli
    "main"          : "cli",
    # evaluator
    "EvaluateConfig": "evaluator",
    "evaluate"      : "evaluator",
    # predictor
    "PredictConfig" : "predictor",
    "predict"       : "predictor",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "EvaluateConfig",
    "PredictConfig",
    "evaluate",
    "main",
    "predict",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.runners.cli import main
    from mon.runners.evaluator import EvaluateConfig, evaluate
    from mon.runners.predictor import PredictConfig, predict
