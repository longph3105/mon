"""Tests for ``mon.core.type_aliases``."""

import typing

import numpy as np
import torch

from mon.core.type_aliases import Float3, Int2, Int3, OneOrPair, TensorOrArray


def test_tensor_or_array_isinstance():
    assert isinstance(torch.zeros(1), TensorOrArray)
    assert isinstance(np.zeros(1), TensorOrArray)
    assert not isinstance([0.0], TensorOrArray)


def test_tuple_aliases():
    assert typing.get_args(Int2) == (int, int)
    assert typing.get_args(Int3) == (int, int, int)
    assert typing.get_args(Float3) == (float, float, float)


def test_one_or_pair_is_generic():
    alias = OneOrPair[int]
    assert typing.get_origin(alias) is OneOrPair
    assert typing.get_args(alias) == (int,)
    assert OneOrPair.__value__ is not None  # lazily evaluated; must resolve without error
