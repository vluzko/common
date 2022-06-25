import numpy as np
import torch
import torchtext
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType


class OneLayerAttn(nn.Module):
    """A single attention only layer module"""

    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert self.model % self.n_head == 0
        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # embedding
        # attention
        # mat mul
        # linear out
        raise NotImplementedError


class PosEncode(nn.Module):
    """Positional encoding"""

    def __init__(self, k, d) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Model(nn.Module):
    """Wrapper around one layer attention"""

    def __init__(self, vocab_size: int, d_model: int=256) -> None:
        super().__init__()
        self.one_layer = OneLayerAttn(d_model, 8)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PosEncode(d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(inputs)
        # Positional encoding
        encoded = self.pos(embedded) + embedded
        return self.one_layer(encoded)


def load_data():
    # Load some text dataset
    data_iter = torchtext.datasets.IMDB(split='train')
    return data_iter


def train():
    model = Model()
    opt = optim.Adam(model.parameters())
    data = load_data()

    for i, (data, target) in enumerate(data):
        # Masking
        output = model(data)
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        opt.step()
        opt.zero_grad()

    return model


def analyze():
    # Run models
    # Freeze attention layer
    raise NotImplementedError


def nat_contrib():
    """Calculate the information contribution of each head and circuit"""
    raise NotImplementedError


def main():
    train()


if __name__ == "__main__":
    main()
