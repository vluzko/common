import torch

from torch import nn, optim, f as F


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        