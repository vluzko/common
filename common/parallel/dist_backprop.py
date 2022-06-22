import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from typing import List


class MLP(nn.Module):
    """An MLP"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=128, num_layers: int=3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_widths: List[int] = [input_dim, *[hidden_dim]*(num_layers-1)]
        self.layers = nn.Sequential(*(x for i in range(num_layers-1) for x in 
            (nn.Linear(self.layer_widths[i], self.layer_widths[i+1]), nn.ReLU()
        )))
        self.final = nn.Linear(self.layer_widths[-1], output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.final(self.layers(inputs))


def train():
    # Build model
    # Distribute model
    # Send data to models
    # Get gradients back
    # Combine gradients
    # Send updates to GPUs
    raise NotImplementedError
    

if __name__ == "__main__":
    train()
