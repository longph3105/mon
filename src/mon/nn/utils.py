"""Model helpers: parameter counts, freezing, device lookup."""

__all__ = [
    "count_parameters",
    "freeze",
    "module_device",
    "unfreeze",
]

import torch
from torch import nn


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Number of parameters (shared parameters counted once).

    Examples:
        >>> count_parameters(nn.Conv2d(3, 8, 3))
        224
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def freeze(module: nn.Module) -> nn.Module:
    """Stop gradients for all parameters of ``module``; returns it.

    Doesn't change train/eval mode: call ``.eval()`` too if BatchNorm statistics or
    dropout should be frozen.
    """
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def unfreeze(module: nn.Module) -> nn.Module:
    """Re-enable gradients for all parameters of ``module``; returns it."""
    for p in module.parameters():
        p.requires_grad_(True)
    return module


def module_device(module: nn.Module) -> torch.device:
    """The device holding ``module``'s parameters and buffers.

    Raises:
        ValueError: If the module has no parameters or buffers, or they are spread over
            several devices (pass inputs per device yourself then).

    Examples:
        >>> module_device(nn.Linear(2, 2))
        device(type='cpu')
    """
    devices = {t.device for t in (*module.parameters(), *module.buffers())}
    if not devices:
        raise ValueError(f"{type(module).__name__} has no parameters or buffers to take a device from")
    if len(devices) > 1:
        raise ValueError(f"{type(module).__name__} is spread over several devices: {sorted(map(str, devices))}")
    return devices.pop()
