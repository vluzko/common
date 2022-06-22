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
        mean = inputs.mean(dim=-1)
        normalized = (inputs-mean) / (torch.sqrt(inputs.var(dim=-1)) + 1e-6)
        return normalized


class MyBatchNorm(nn.Module):
    """My batch norm"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError