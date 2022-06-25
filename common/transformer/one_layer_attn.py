import numpy as np
import torch
import torchtext
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType


class OneLayerAttn(nn.Module):
    """A single attention only layer module"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # embedding
        # attention
        # mat mul
        # linear out
        raise NotImplementedError


def load_data():
    # Load some text dataset
    raise NotImplementedError


def train():
    # Make model
    # Make optim
    # Run through dataset
    # Update with bce
    # Return model
    raise NotImplementedError


def analyze():
    # Run models
    # Freeze attention layer
    raise NotImplementedError


def nat_contrib():
    """Calculate the information contribution of each head and circuit"""
    raise NotImplementedError


def main():
    data_iter = torchtext.datasets.IMDB(split='train')
    raise NotImplementedError


if __name__ == "__main__":
    main()
