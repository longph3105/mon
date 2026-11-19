"""Training recipes: a model's step, loss, init, and official defaults.

A model module defines one ``TrainRecipe`` per trainable model and names it in the model's
``MODELS`` entry (``train="module:RECIPE"``); ``mon.runners.trainer`` runs the generic loop
around it. The recipe's ``defaults`` reproduce the official training script and sit under
the user's config: every key stays overridable.

Most recipes use one optimizer over every parameter. A recipe that alternates optimizers
(SCI++) names them in ``optimizers`` (each over some submodules, all built from the config's
``optimizer``) and picks one per step with ``schedule``; its step returns ``loss_<name>`` for
each optimizer, and the runner backpropagates the scheduled one with only that group's
parameters requiring grad.
"""

__all__ = [
    "StepOutput",
    "TrainRecipe",
]

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

type StepOutput = dict[str, Tensor]
"""``{"loss": scalar to backpropagate, <term>: scalar, …}``; every value is logged (detached)."""


@dataclass(frozen=True)
class TrainRecipe:
    """One model's training definition.

    Attributes:
        loss: Builds the loss module (moved to the training device by the runner).
        step: ``(model, loss, batch) -> StepOutput`` for one training batch; ``batch`` is the
            collated dict (``"image"``, optional ``"target"``, ``"meta"``), already on the device.
            The model is in train mode; the runner does backward, clipping, and the optimizer step.
        init: Applied to a freshly built model (training from scratch) after seeding, when the
            official script re-initializes parts the constructor doesn't; ``None``: constructor init.
        defaults: ``TrainConfig`` values of the official recipe (``optimizer``, ``epochs``,
            ``batch_size``, ``resize``, ``grad_clip``, ``seed``, …), applied under the user's config.
        source: The official script these follow, for the record.
        notes: Known differences from the official script.
        optimizers: ``{name: (submodule attribute, …)}``: one optimizer per entry over those
            submodules' parameters (disjoint); ``None``: one optimizer over all parameters.
        schedule: With ``optimizers``: ``(step) -> name`` of the optimizer that updates at this
            1-based global step, or ``None`` for a step that updates nothing (the batch is
            consumed without a forward pass). The step function then returns ``loss_<name>``
            for every optimizer; the scheduled one is backpropagated and logged as ``loss``.
    """

    loss      : Callable[[], nn.Module]
    step      : Callable[[nn.Module, nn.Module, dict[str, Any]], StepOutput]
    init      : Callable[[nn.Module], None] | None = None
    defaults  : Mapping[str, Any] = field(default_factory=dict)
    source    : str = ""
    notes     : str = ""
    optimizers: Mapping[str, tuple[str, ...]] | None = None
    schedule  : Callable[[int], str | None] | None = None

    def __post_init__(self) -> None:
        if (self.optimizers is None) != (self.schedule is None):
            raise ValueError("TrainRecipe: optimizers and schedule go together")
        if self.optimizers is not None:
            parts = [p for group in self.optimizers.values() for p in group]
            if not self.optimizers or not all(self.optimizers.values()) or len(parts) != len(set(parts)):
                raise ValueError(f"TrainRecipe.optimizers must be non-empty, disjoint groups, got {dict(self.optimizers)}")
