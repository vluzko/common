from typing import Tuple
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType
import torchvision
from torchvision import transforms

from common import config


class Encoder(nn.Module):
    """VAE Encoder"""

    def __init__(self, input_size: int, z_width: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.z_width = z_width
        self.layers = nn.Sequential(*[
            nn.Conv2d(1, 16, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 16, 4),
            nn.ReLU(),
        ])
        self.mean_layer = nn.Conv2d(16, 1, 4)
        self.var_layer = nn.Conv2d(16, 1, 4)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor. torch.Tensor]:
        x = self.layers(inputs)
        return self.mean_layer(x), self.var_layer(x)


class Decoder(nn.Module):
    """VAE Decoder"""

    def __init__(self, output_size: int, z_width: int) -> None:
        super().__init__()
        self.output_size = output_size
        self.z_width = z_width
        self.layer_1 = nn.Linear(z_width, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer_1(inputs)


class VAE(nn.Module):
    """A variational autoencoder"""

    def __init__(self, space_size: int, z_width: int) -> None:
        super().__init__()
        self.z_width = z_width
        self.encoder = Encoder(space_size, z_width)
        self.decoder = Decoder(space_size, z_width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        space = torch.distributions.Normal(encoded[:, 0], encoded[:, 1])
        sample = space.rsample()
        decoded = self.decoder(sample)
        return decoded


def train(dataset: DataLoader, vae: VAE, lr: float=0.001):
    opt = optim.Adam(vae.parameters(), lr=lr)

    for i, (inputs, target) in enumerate(dataset):
        pred = vae(inputs)
        # loss = ?
        loss.backward()
        opt.step()
        opt.zero_grad()

        print(loss)

    return vae


if __name__ == '__main__':
    root_dir = config.DATA / 'mnist'
    root_dir.mkdir(exist_ok=True)
    model = VAE(5, 16)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(str(root_dir), train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(str(root_dir), train=False, download=True, transform=transform)
    train(train_set, model)
