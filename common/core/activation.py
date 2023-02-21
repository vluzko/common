import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader


def relu(inputs: torch.Tensor) -> torch.Tensor:
    return functional.threshold(inputs, 0.0, 0.0)


def swish(inputs: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return inputs / (1.0 + torch.exp(-beta * inputs))


def swiglu(inputs: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError