from typing import List, Tuple
import numpy as np
import torch
import torchtext
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType


def vocabularize(dataset: List[Tuple[str, str]]):
    """Take input/target pairs, produce a vocabulary and one hot everything"""