import torch
from torch import nn
from time import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class NoOpt(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[nn.Linear(10, 10) for _ in range(10)])

    def forward(self, x):
        return self.layers(x)



class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        new_h = torch.tanh(self.linear(x))
        return new_h


def no_opt():
    model = NoOpt()
    model.to(DEVICE)
    for _ in range(1000):
        x = torch.rand(10, 10).to(DEVICE)
        model(x)


# opt_model = torch.jit.trace(MyCell(), torch.rand(10, 10))
# @torch.jit.script
# def with_opt():
#     # opt_model.to(DEVICE)
#     for _ in range(1000):
#         x = torch.rand(10, 10)
#         opt_model(x)


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
print(traced_cell)
traced_cell(x, h)

# start1 = time()
# no_opt()
# end1 = time()
# print(end1 - start1)

# start2 = time()
# with_opt()
# end2=time()
# print(end2 - start2)