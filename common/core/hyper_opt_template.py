from typing import Tuple
import optuna
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType
from einops import rearrange


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    """A model"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def generate_data(num_points: int=10000) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand((num_points, 8), device=DEVICE) * 100 - 50
    outputs = torch.cos(inputs) + torch.sin(inputs)
    return inputs, outputs


def ray_tune():
    raise NotImplementedError


def run_optuna():
    data = generate_data()
    test_data = generate_data()

    def train(trial) -> float:
        batch_size = trial.suggest_int('batch_size', 4, 128)
        lr = trial.suggest_float('learning_rate', 1e-6, 1e-3)
        model = Model().to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(*data)
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)

        for epoch in range(10):
            for i, o in loader:
                out = model(i)
                loss = functional.mse_loss(out, o)
                loss.backward()
                opt.step()
                opt.zero_grad()

        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*test_data), shuffle=True)

        total_loss = 0.0
        with torch.no_grad():
            for i, o in test_loader:
                out = model(i)
                loss = functional.mse_loss(out, o)
        mean_loss = total_loss / len(test_data[0])

        return mean_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(train, n_trials=10)


if __name__ == "__main__":
    run_optuna()


