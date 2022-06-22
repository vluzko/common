import numpy as np
import torch
from torchtyping import TensorType  # type: ignore
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader


def relu(inputs: TensorType["batch", "n"]) -> TensorType["batch", "n"]:
    raise NotImplementedError