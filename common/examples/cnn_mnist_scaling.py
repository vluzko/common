import numpy as np
import torch
import torchvision
from torch import cuda, nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType
from einops import rearrange

from common import config


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN(nn.Module):
    """CNN"""

    def __init__(self, input_size: int, num_cats: int, num_channels: int=16) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_cats = num_cats

        self.layers = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3),
            nn.MaxPool2d(2),
            nn.Flatten(1, 3)
        )
        # I'm not calculating this shit analytically
        with torch.no_grad():
            sample_input = torch.randn((2, 1, input_size, input_size))
            sample_output = self.layers(sample_input)

        self.output = nn.Linear(sample_output.shape[-1], self.num_cats)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.layers(inputs)
        return functional.softmax(self.output(x), dim=0)


def train(model, train_loader, lr: float=0.001, epochs:int = 50):
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            pred = model(inputs)
            loss = functional.binary_cross_entropy(pred, functional.one_hot(targets, 10).float())
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss)
    return loss


def test():
    raise NotImplementedError


def main():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    train_data, valid_data = torch.utils.data.random_split(torchvision.datasets.MNIST(root=config.DATA / 'mnist', train=True, download=True, transform=transforms), [50000, 10000])
    # test_dataset = torchvision.datasets.MNIST(root=config.DATA / 'mnist', train=True, download=True, transform=transforms)

    for batch_size in (4, 8, 16, 32, 64):
        model = CNN(28, 10).to(DEVICE)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        for lr in torch.logspace(1e-5, 1e-2, steps=10):
            print(train(model, train_loader, lr=lr))


if __name__ == '__main__':
    main()