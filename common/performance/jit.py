import torch
from torch import nn
from torch.nn import functional
from time import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NoOpt(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.layer(x) + h)
        return new_h, new_h



class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        new_h = torch.tanh(self.linear(x))
        return new_h


class MyCell2(torch.nn.Module):
    def __init__(self):
        super(MyCell2, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell2()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))

batch_size = 256
n = 10000
start1 = time()
opt = torch.optim.Adam(traced_cell.parameters())
for i in range(n):
    x, h = torch.rand(batch_size, 4), torch.rand(batch_size, 4)

    new_x, new_h = traced_cell(x, h)
    loss = functional.mse_loss(new_h, h)
    loss.backward()
    opt.step()
    opt.zero_grad()
end1 = time()
print(end1 - start1)


no_opt = MyCell2()
start2 = time()
opt1 = torch.optim.Adam(no_opt.parameters())
for i in range(n):
    x, h = torch.rand(batch_size, 4), torch.rand(batch_size, 4)
    new_x, new_h = no_opt(x, h)
    loss = functional.mse_loss(new_h, h)
    loss.backward()
    opt.step()
    opt.zero_grad()

end2=time()
print(end2 - start2)