from typing import Callable
import torch
from torch.nn import functional

from common.core import activation


def check_activation(c: Callable, torch_version: Callable):
    t = torch.randn(size=(10, 10))
    mine = c(t)
    theirs = torch_version(t)
    assert torch.isclose(mine, theirs).all()


def test_relu():
    check_activation(activation.relu, torch.relu)


def test_swish():
    check_activation(activation.swish, functional.silu)


def test_swiglu():
    raise NotImplementedError