"""Tests for ``mon.nn.utils``."""

import pytest
import torch
from torch import nn

from mon.nn.utils import count_parameters, freeze, module_device, unfreeze


def test_count_and_freeze():
    shared = nn.Linear(4, 4)
    model  = nn.Sequential(shared, nn.ReLU(), shared, nn.Conv2d(3, 2, 1))  # shared layer counted once
    assert count_parameters(model) == 20 + 8
    freeze(model[3])
    assert count_parameters(model, trainable_only=True) == 20
    assert unfreeze(freeze(model)) is model and count_parameters(model, trainable_only=True) == 28


def test_freeze_keeps_mode():
    model = nn.BatchNorm2d(3).train()
    freeze(model)
    assert model.training


def test_module_device():
    assert module_device(nn.Linear(2, 2)) == torch.device("cpu")
    bn = nn.BatchNorm2d(2, affine=False)  # buffers only
    assert module_device(bn) == torch.device("cpu")
    with pytest.raises(ValueError, match="no parameters"):
        module_device(nn.ReLU())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs a second device (MPS)")
def test_module_device_mixed():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2).to("mps"))
    with pytest.raises(ValueError, match="several devices"):
        module_device(model)
