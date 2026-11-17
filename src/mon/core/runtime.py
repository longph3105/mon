"""Runtime helpers: device discovery and selection, and global RNG seeding.

Nothing here runs at import; every call reads the current system state.
"""

__all__ = [
    "available_devices",
    "seed_everything",
    "select_device",
]

import os
import random
import secrets

import numpy as np
import torch

from mon.core.console import get_logger

logger = get_logger(__name__)

_MAX_SEED           = 2**32 - 1          # NumPy's legacy seeding requires seed < 2**32.
_CUBLAS_ENV         = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_DETERMINISM = ":4096:8"          # Value from the CUDA/PyTorch reproducibility docs.


# --- Devices ---

def available_devices() -> list[str]:
    """List usable devices: ``"cpu"``, then ``"mps"`` if present, then ``"cuda:0"``, ….

    CUDA indices follow torch (they respect ``CUDA_VISIBLE_DEVICES``).
    """
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
    return devices


def select_device(spec: str | int | torch.device | None = "auto") -> torch.device:
    """Return a ``torch.device`` for ``spec``, raising if it isn't available.

    Args:
        spec: ``"auto"``/``None`` (first of ``cuda:0``, ``mps``, ``cpu``), ``"cpu"``,
            ``"mps"``, ``"cuda"`` (= ``"cuda:0"``), ``"cuda:N"``, an int ``N``
            (= ``"cuda:N"``), or a ``torch.device``. Strings are case-insensitive.

    Raises:
        ValueError: If the device is unknown or unavailable; the message lists the
            available devices. There is never a silent fallback to CPU.
        TypeError: If ``spec`` has an unsupported type.
    """
    available = available_devices()

    if spec is None:
        name = "auto"
    elif isinstance(spec, bool):
        raise TypeError(f"device spec must be str, int, or torch.device, got {spec!r}")
    elif isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"CUDA device index must be >= 0, got {spec}")
        name = f"cuda:{spec}"
    elif isinstance(spec, torch.device):
        name = str(spec)
    elif isinstance(spec, str):
        name = spec.strip().lower()
    else:
        raise TypeError(f"device spec must be str, int, or torch.device, got {type(spec).__name__}")

    if name == "auto":
        for preferred in ("cuda:0", "mps"):
            if preferred in available:
                return torch.device(preferred)
        return torch.device("cpu")

    name = {"cuda": "cuda:0", "mps:0": "mps", "cpu:0": "cpu"}.get(name, name)
    if name not in available:
        raise ValueError(f"device {spec!r} is not available; available devices: {', '.join(available)}")
    return torch.device(name)


# --- Reproducibility ---

def seed_everything(
    seed         : int | None = None,
    *,
    deterministic: bool = False,
    warn_only    : bool = False,
) -> int:
    """Seed Python ``random``, NumPy's global RNG, and torch (CPU, CUDA, MPS).

    The single seeding entry point for ``mon``. Log or save the returned seed with the
    run so it can be reproduced.

    Args:
        seed: Seed in ``[0, 2**32 - 1]``. ``None`` draws a random one and logs it.
        deterministic: Also make torch use deterministic algorithms: sets
            ``cudnn.deterministic``, disables ``cudnn.benchmark``, sets
            ``CUBLAS_WORKSPACE_CONFIG`` if unset, and calls
            ``torch.use_deterministic_algorithms(True)``. Passing ``False`` does not
            undo an earlier ``True``.
        warn_only: With ``deterministic``, warn instead of raising when an op has no
            deterministic implementation.

    Returns:
        The seed used.

    Raises:
        TypeError: If ``seed`` isn't an int.
        ValueError: If ``seed`` is out of range.

    Notes:
        ``PYTHONHASHSEED`` only takes effect when set before Python starts, so set it in
        the shell or launcher if ``hash()`` ordering matters. ``CUBLAS_WORKSPACE_CONFIG``
        must be set before CUDA initializes; call this early.
    """
    if seed is None:
        seed = secrets.randbelow(_MAX_SEED + 1)
        logger.info("seed_everything: no seed given, using seed=%d", seed)
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {seed!r}")
    elif not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Safe without CUDA (no-op / deferred).
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        if _CUBLAS_ENV not in os.environ:
            if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]  # unannotated in torch
                logger.warning("%s set after CUDA initialized; cuBLAS may stay non-deterministic", _CUBLAS_ENV)
            os.environ[_CUBLAS_ENV] = _CUBLAS_DETERMINISM
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return seed
