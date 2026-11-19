"""Shared fixtures."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture
def regression_device() -> Iterator[torch.device]:
    """CUDA if available, else CPU, with TF32 off so CUDA matches CPU float32 within the regression tolerances."""
    if not torch.cuda.is_available():
        yield torch.device("cpu")
        return
    saved = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield torch.device("cuda")
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved
