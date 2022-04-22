from typing import Tuple
import torch
from torch import nn


class Module:

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def cust_back(self, *inputs: torch.Tensor) -> torch.Tensor:
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
        self._last_activation = input
        output = torch.matmul(input, self.weight)

        return output + self.bias

    def cust_back(self, next_err: torch.Tensor) -> torch.Tensor:  # type: ignore
        import pdb
        pdb.set_trace()
        return self._last_activation * next_err


class ReLU(Module):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._last_activation = input
        return input * (input > 0).float()

    def cust_back(self, next_weight: torch.Tensor, next_err: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Equation: W_{l+1}^T * error hadamard derivative(ReLU)(last_act)
        """
        # grad_act = torch.ones_like(self._last_activation) * (self._last_activation > 0).float()
        act_deriv = self._last_activation > 0
        grad_act = next_weight.transpose(0, 1) * next_err * act_deriv
        return grad_act


class MSELoss(Module):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._last_activation = input
        self._target = target
        return torch.mean((input - target) ** 2)

    def cust_back(self):
        grad_act = 2 * (self._last_activation - self._target)
        return grad_act


class Network(Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_one = Linear(input_dim, hidden_dim)
        self.act = ReLU()
        self.layer_two = Linear(hidden_dim, output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.layer_one.forward(input)
        output = self.act.forward(output)
        output = self.layer_two.forward(output)
        return output

    def cust_back(self, loss_back: torch.Tensor):  # type: ignore
        sec_layer = self.layer_two.cust_back(loss_back)
        act_back = self.act.cust_back(self.layer_two.weight, loss_back)
        first_layer = self.layer_one.cust_back(act_back)
        return first_layer, sec_layer

    def update(self, weights, lr: float=0.0001):
        w1, w2 = weights
        self.layer_one.weight -= lr * w1
        self.layer_two.weight -= lr * w2


def make_data(size: int=1000) -> Tuple[torch.Tensor, torch.Tensor]:
    ins = torch.arange(size * 2).float().view(size, 2)
    outs = torch.sin(ins)
    outs[:, 1] = torch.cos(ins[:, 0])
    return ins, outs


def train():
    net = Network(input_dim=2, hidden_dim=2, output_dim=1)
    loss = MSELoss()
    ins, outs = make_data()

    with torch.no_grad():
        for i, (x, y) in enumerate(zip(ins, outs)):
            output = net.forward(x)
            loss_val = loss.forward(x, y)
            loss_back = loss.cust_back()
            updates = net.cust_back(loss_back)
            net.update(updates)
            print(i)


if __name__ == '__main__':
    train()