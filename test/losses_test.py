import torch
from torch.nn import functional
from common.core import losses


def test_nll():
    K = 5
    x = functional.log_softmax(torch.randn(K))
    y = torch.zeros(K)
    y[3] = 1
    mine = losses.nll_loss(x, y)
    theirs = functional.nll_loss(x, y)