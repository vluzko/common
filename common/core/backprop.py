import torch
from torch import nn


class Module:

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def backward(self) -> torch.Tensor:
        raise NotImplementedError


class Linear(Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.zeros(self.input_dim, self.output_dim, requires_grad=False)
        self.weight = nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = torch.zeros(self.output_dim, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight)

        return output + self.bias

    def backward(self):
        pass


class ReLU(Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.maximum(input, torch.zeros_like(input))

    def backward(self):
        raise NotImplementedError


class MSELoss(Module):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((input - target) ** 2)

    def backward(self):
        raise NotImplementedError