from typing import List
import numpy as np
import torch
from torch import nn, optim, multiprocessing as mp
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader

def f(i: int, weight, inputs):
            dev = f'cuda:{i}'
            sub_weight = weight.to(dev)
            layer_out = inputs.to(dev) @ sub_weight
            return layer_out.cpu()


class RowLinear(nn.Module):
    """A linear layer with row parallelism"""

    def __init__(self, in_size: int, out_size: int, n_devices: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_devices = n_devices
        self.device_size = in_size // n_devices

        assert out_size % n_devices == 0

        self.weight = nn.Parameter(torch.empty((n_devices, self.device_size, out_size)))
        self.bias = nn.Parameter(torch.zeros(out_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')


    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs = inputs.view(self.n_devices, -1)
        pool = mp.Pool(self.n_devices)
        
        gathered = pool.map(f, zip(range(self.n_devices), self.weight, inputs))
        # for i in range(self.n_devices):
        #     dev = f'cuda:{i}'
        #     sub_weight = self.weight[i].to(dev)
        #     layer_out = inputs[i].to(dev) @ sub_weight
        #     gathered.append(layer_out.cpu())
        joined = torch.sum(torch.stack(gathered), dim=0)
        return joined + self.bias


class ColumnLinear(nn.Module):
    """A linear layer with column parallelism"""

    def __init__(self, in_size: int, out_size: int, n_devices: int) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.n_devices = n_devices
        self.device_size = out_size // n_devices

        assert out_size % n_devices == 0

        self.weight = nn.Parameter(torch.empty((n_devices, in_size, out_size // n_devices)))
        self.bias = nn.Parameter(torch.zeros((n_devices, out_size // n_devices)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        gathered = [] 
        for i in range(self.n_devices):
            dev = f'cuda:{i}'
            sub_weight = self.weight[i].to(dev)
            sub_bias = self.bias[i].to(dev)
            layer_out = inputs.to(dev) @ sub_weight + sub_bias
            gathered.append(layer_out)

        return gathered


class TensorMLP(nn.Module):
    """Tensor parallel MLP"""

    def __init__(self, model_dim: int, n_devices: int) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.n_devices = n_devices

        self.l1 = ColumnLinear(model_dim, 4* model_dim, n_devices)
        self.l2 = RowLinear(model_dim*4, model_dim, n_devices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = [functional.relu(y) for y in x]  # type: ignore
        return self.l2(x)


# model = ColumnLinear(12, 12, 4)
# x = torch.arange(12).float()
# y = torch.arange(12).float()
# out = torch.cat([val.cpu() for val in model(x)])
# loss = functional.mse_loss(out, y)
# loss.backward()
# print(model(x))


model = TensorMLP(12, 4)
opt = optim.Adam(model.parameters())
x = torch.arange(12).float()
y = torch.arange(12).float()
x.share_memory_()
model.share_memory()
losses = []
for _ in range(1000):
    # out = torch.cat([val.cpu() for val in model(x)])
    out = model(x)
    loss = functional.mse_loss(out, y)
    loss.backward()
    opt.step()
    opt.zero_grad()

    losses.append(loss)
import pdb
pdb.set_trace()
