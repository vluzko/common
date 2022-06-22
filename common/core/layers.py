import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader


class MyLayerNorm(nn.Module):
    """My layer norm"""

    def __init__(self) -> None:
        super().__init__()


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1).view(*inputs.shape[:-1], 1)
        var = inputs.var(unbiased=False, dim=-1).view(*inputs.shape[:-1], 1)
        normalized = (inputs-mean) / (torch.sqrt(var + 1e-5))
        return normalized


class MyBatchNorm(nn.Module):
    """My batch norm"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=0)
        var = inputs.var(unbiased=False, dim=0)
        return (inputs - mean) / (torch.sqrt(var + 1e-5))