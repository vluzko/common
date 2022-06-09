from typing import Tuple
import torch
from torch import nn, optim
from torch.nn import functional


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
        self.weight = torch.arange(self.output_dim * self.input_dim, requires_grad=False).float().view(self.output_dim, self.input_dim)
        # self.weight = nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        self.bias = torch.zeros(self.output_dim, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._last_activation = input
        output = torch.matmul(input, self.weight.transpose(0, 1))

        return output + self.bias

    def cust_back(self, next_err: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self._last_activation.view(-1, 1) * next_err


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
        grad_act = next_weight.transpose(0, 1) @ next_err * act_deriv
        return grad_act


class MSELoss(Module):

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore
        self._last_activation = input
        self._target = target
        return torch.mean((input - target) ** 2)

    def cust_back(self):
        grad_act = 2 * (self._last_activation - self._target)
        return 1 / grad_act.shape[0] * grad_act


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
        import pdb
        pdb.set_trace()
        act_back = self.act.cust_back(self.layer_two.weight, loss_back)
        first_layer = self.layer_one.cust_back(act_back)
        return first_layer, sec_layer

    def update(self, weights, lr: float=0.0001):
        w1, w2 = weights
        self.layer_one.weight -= lr * w1.transpose(0, 1)
        self.layer_two.weight -= lr * w2.transpose(0, 1)


class TorchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_one = nn.Linear(input_dim, hidden_dim)
        self.layer_two = nn.Linear(hidden_dim, output_dim)

    def copy_weights(self, other: Network):
        with torch.no_grad():
            self.layer_one.weight.copy_(other.layer_one.weight.detach().clone())
            self.layer_one.bias.copy_(other.layer_one.bias.detach().clone())
            self.layer_two.weight.copy_(other.layer_two.weight.detach().clone())
            self.layer_two.bias.copy_(other.layer_two.bias.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer_one(inputs)
        output = torch.relu(output)
        output = self.layer_two(output)
        return output


def make_data(size: int=1000) -> Tuple[torch.Tensor, torch.Tensor]:
    ins = torch.arange(size * 2).float().view(size, 2)
    outs = torch.sin(ins)
    outs[:, 1] = torch.cos(ins[:, 0])
    return ins, outs


def make_lin_data(size: int=1000, in_space: int=2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ins = torch.arange(size * in_space).float().view(size, -1)
    weight = torch.randn(in_space, 1)
    outs = torch.matmul(ins, weight)
    return ins, outs, weight


def train(size: int=10) -> None:
    loss = MSELoss()
    ins, outs = make_data(size)
    net = Network(input_dim=ins.shape[1], hidden_dim=8, output_dim=outs.shape[1])
    tnet = TorchNet(input_dim=ins.shape[1], hidden_dim=8, output_dim=outs.shape[1])
    tnet.copy_weights(net)

    # with torch.no_grad():
    for i, (x, y) in enumerate(zip(ins, outs)):
        output = net.forward(x)
        toutput = tnet(x)
        loss_val = loss.forward(output, y)
        t_loss = functional.mse_loss(toutput, y)
        t_loss.retain_grad()
        assert torch.isclose(loss_val, t_loss)
        loss_back = loss.cust_back()
        updates = net.cust_back(loss_back)
        t_loss.backward()
        import pdb
        pdb.set_trace()
        net.update(updates)


def linear_train(size: int=100, epochs: int = 1000, lr=1e-5) -> None:
    torch.manual_seed(1)
    loss = MSELoss()
    ins, outs, weight = make_lin_data(size)
    net = Linear(input_dim=ins.shape[1], output_dim=outs.shape[1])
    tnet = nn.Linear(ins.shape[1], outs.shape[1])
    with torch.no_grad():
        tnet.weight.copy_(net.weight)
        tnet.bias.copy_(net.bias)
    opt = optim.SGD(tnet.parameters(), lr=lr)
    for j in range(epochs):
        indices = torch.randperm(ins.shape[0])
        ins = ins[indices]
        outs = outs[indices]
        for i, (x, y) in enumerate(zip(ins, outs)):
            output = net.forward(x)
            tout = tnet(x)
            # toutput = tnet(x)
            loss_val = loss.forward(output, y)
            t_loss = functional.mse_loss(tout, y)
            # assert torch.isclose(t_loss, loss_val)
            loss_back = loss.cust_back()
            updates = net.cust_back(loss_back)
            net.weight -= lr * updates.transpose(0, 1)
            t_loss.backward()
            opt.step()
            opt.zero_grad()
        print(t_loss.item())
        print(loss_val.item())
        lr *= 0.99


def relu_linear_train(size: int=100, epochs: int = 1000, lr=1e-5):
    torch.manual_seed(1)
    loss = MSELoss()
    ins, outs, weight = make_lin_data(size)
    net = Linear(input_dim=ins.shape[1], output_dim=outs.shape[1])
    act = ReLU()
    tnet = nn.Linear(ins.shape[1], outs.shape[1])
    tact = nn.ReLU()
    with torch.no_grad():
        tnet.weight.copy_(net.weight)
        tnet.bias.copy_(net.bias)
    opt = optim.SGD(tnet.parameters(), lr=lr)
    for j in range(epochs):
        indices = torch.randperm(ins.shape[0])
        ins = ins[indices]
        outs = outs[indices]
        for i, (x, y) in enumerate(zip(ins, outs)):
            output = act(net.forward(x))
            tout = tact(tnet(x))
            # toutput = tnet(x)
            loss_val = loss.forward(output, y)
            t_loss = functional.mse_loss(tout, y)
            # assert torch.isclose(t_loss, loss_val)
            loss_back = loss.cust_back()
            updates = net.cust_back(loss_back)
            net.weight -= lr * updates.transpose(0, 1)
            t_loss.backward()
            opt.step()
            opt.zero_grad()
        print(t_loss.item())
        print(loss_val.item())
        lr *= 0.99
    raise NotImplementedError


if __name__ == '__main__':
    linear_train()