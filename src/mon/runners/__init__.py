"""Runners: drive models over data end to end: training, prediction, evaluation.

Names are re-exported lazily (see ``mon.core``).

Submodules (kept in sync with the code by ``tests/test_namespaces.py``):

    =========  ===========================================================================  ========================
    Module     Purpose                                                                      Key names
    =========  ===========================================================================  ========================
    cli        Command-line entry point: ``mon train``, ``mon predict``, ``mon evaluate``.  main
    evaluator  Score saved predictions against targets with image-quality metrics.          EvaluateConfig, evaluate
    predictor  Run a model over images and save its outputs.                                PredictConfig, predict
    trainer    Train a registered model with its recipe, validate, checkpoint, and resume.  TrainConfig, train
    =========  ===========================================================================  ========================
"""

from typing import TYPE_CHECKING

from mon.core._lazy import make_lazy

_SUBMODULES: tuple[str, ...] = ("cli", "evaluator", "predictor", "trainer")

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
    # trainer
    "OptimizerConfig"     : "trainer",
    "SchedulerConfig"     : "trainer",
    "TrainConfig"         : "trainer",
    "resolve_train_config": "trainer",
    "train"               : "trainer",
}

# Literal list so static tools (mypy, pyright) see the re-exports; must equal sorted(_EXPORTS).
__all__ = [
    "EvaluateConfig",
    "OptimizerConfig",
    "PredictConfig",
    "SchedulerConfig",
    "TrainConfig",
    "evaluate",
    "main",
    "predict",
    "resolve_train_config",
    "train",
]

__getattr__, __dir__ = make_lazy(__name__, _EXPORTS, _SUBMODULES, globals())


if TYPE_CHECKING:  # Real imports for type checkers and IDEs; never executed.
    from mon.runners.cli import main
    from mon.runners.evaluator import EvaluateConfig, evaluate
    from mon.runners.predictor import PredictConfig, predict
    from mon.runners.trainer import OptimizerConfig, SchedulerConfig, TrainConfig, resolve_train_config, train
