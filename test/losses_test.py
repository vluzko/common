import torch
from torch.nn import functional
from common.core import losses


def test_nll():
    k = 5
    batch = 3
    x = functional.log_softmax(torch.randn(batch, k), dim=1)
    y = torch.randint(0, k, (batch, 1))
    mine = losses.nll_loss(x, y)
    theirs = functional.nll_loss(x, y.view(batch))
    assert torch.isclose(mine, theirs)


def kl_loss():
    k = 5
    batch = 3
    x = torch.randn(batch, k)
    